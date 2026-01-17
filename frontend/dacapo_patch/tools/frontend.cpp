#include <algorithm>
#include <atomic>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <sys/stat.h>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/SourceMgr.h"
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Dialect/Earth/Transforms/Passes.h"

#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

using namespace mlir;

void handler(int sig) {
  void *array[10];
  size_t size;

  size = backtrace(array, 10);

  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

namespace hecate {

using valueID = size_t;
using funcID = size_t;

struct Context {
  Context();
  mlir::MLIRContext ctxt;
  mlir::OwningOpRef<mlir::ModuleOp> mod;
  std::unique_ptr<mlir::OpBuilder> builder;
  llvm::SmallVector<mlir::Value, 32> valueMap;
  llvm::SmallVector<mlir::func::FuncOp, 32> funcMap;
  std::vector<std::string> pendingComments;
  size_t commentCount = 0;
};

Context::Context() : ctxt(), mod(), builder() {

  ctxt.getOrLoadDialect<mlir::func::FuncDialect>();
  auto ed = ctxt.getOrLoadDialect<hecate::earth::EarthDialect>();
  (void)ed; // Suppress unused variable warning

  auto tmp = std::make_unique<mlir::OpBuilder>(&ctxt);
  builder.swap(tmp);

  mod = mlir::OwningOpRef<mlir::ModuleOp>(
      mlir::ModuleOp::create(builder->getUnknownLoc()));
}

extern "C" {

valueID createConstant(Context *ctxt, double *data, int64_t len, char *filename,
                       size_t line) {
  auto &&builder = *ctxt->builder;

  auto cons = builder.create<hecate::earth::ConstantOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      llvm::ArrayRef(data, len));

  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}

funcID createFunc(Context *ctxt, char *name, int *inputTys, size_t len,
                  char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&funcMap = ctxt->funcMap;
  llvm::SmallVector<mlir::Type, 4> arg_types(len);
  std::transform(inputTys, inputTys + len, arg_types.begin(), [&](auto a) {
    return mlir::RankedTensorType::get(
        llvm::SmallVector<int64_t, 1>{1},
        builder.getType<hecate::earth::CipherType>(0, 0));
  });
  auto funcType = builder.getFunctionType(
      arg_types, mlir::RankedTensorType::get(
                     llvm::SmallVector<int64_t, 1>{1},
                     builder.getType<hecate::earth::CipherType>(0, 0)));
  auto funcOp = mlir::func::FuncOp::create(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0),
      std::string("_hecate_") + name, funcType);
  funcMap.push_back(funcOp);
  ctxt->mod->push_back(funcOp);
  return funcMap.size() - 1;
}

void initFunc(Context *ctxt, funcID fun, valueID *args, size_t len) {
  auto &&funcOp = ctxt->funcMap[fun];
  auto &&valueMap = ctxt->valueMap;
  auto entryBlock = funcOp.addEntryBlock();
  auto funcInput = entryBlock->getArguments();
  ctxt->builder->setInsertionPointToStart(entryBlock);
  {
    int i = 0;
    for (auto a : funcInput) {
      valueMap.push_back(a);
      args[i++] = valueMap.size() - 1;
    }
  }
}

char *save(Context *c, char *const_name, char *mlir_name) {
  c->mod->getOperation()->setAttr(mlir::SymbolTable::getSymbolAttrName(),
                                  c->builder->getStringAttr(mlir_name));
  std::string s_const_name(const_name);
  mlir::PassManager pm(&c->ctxt);
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      earth::createElideConstant({s_const_name + "/"}));
  pm.addNestedPass<func::FuncOp>(earth::createPrivatizeConstant());
  pm.addPass(createCanonicalizerPass());

  auto ret = pm.run(*c->mod);

  size_t optimizedOpCount = 0;
  size_t optimizedConstantCount = 0;
  c->mod->walk([&](mlir::Operation *op) {
    if (!mlir::isa<mlir::ModuleOp>(op) && !op->hasAttr("is_comment_marker")) {
      optimizedOpCount++;
      if (mlir::isa<hecate::earth::ConstantOp>(op)) {

        // Cast to constant operation and extract data
        auto constOp = mlir::cast<hecate::earth::ConstantOp>(op);

        // Access the official data attribute which should survive optimization
        if (auto dataAttr = constOp->getAttrOfType<mlir::ArrayAttr>("data")) {
          size_t dataSize = dataAttr.size();

          // Create the output directory if it doesn't exist
          const char *outdir = "constants";
          mkdir(outdir, 0777);

          // Create filename for this constant inside the folder
          char filename[256];
          snprintf(filename, sizeof(filename), "%s/constant_%zu.txt", outdir,
                   optimizedConstantCount);

          FILE *file = fopen(filename, "w");
          if (file) {

            // Extract all values from ArrayAttr
            std::vector<double> valuesVec;
            for (size_t i = 0; i < dataSize; i++) {
              if (auto floatAttr = dataAttr[i].dyn_cast<mlir::FloatAttr>()) {
                valuesVec.push_back(floatAttr.getValueAsDouble());
              } else {
                valuesVec.push_back(0.0);
              }
            }

            // Write the number of data values as the first line
            fprintf(file, "%zu\n", valuesVec.size());

            // Write the value (i.e. the index) as the second line
            if (auto valueAttr =
                    constOp->getAttrOfType<mlir::IntegerAttr>("value")) {
              fprintf(file, "%lld\n", valueAttr.getValue().getSExtValue());
            } else {
              fprintf(file, "NONE\n");
            }

            // Write the actual data values to the file
            for (size_t i = 0; i < valuesVec.size(); i++) {
              fprintf(file, "%.17g", valuesVec[i]);
              if (i < valuesVec.size() - 1) {
                fprintf(file, "\n");
              }
            }

            fclose(file);
            optimizedConstantCount++;
          }
        } else {
          printf("    No 'data' attribute found! Data may have been processed "
                 "by optimization.\n");
        }
      }
    }
  });

  printf("Operation count (unoptimized): %zu\n", c->valueMap.size());
  printf("Operation count (optimized): %zu\n", optimizedOpCount);
  printf("Constant count (optimized): %zu\n", optimizedConstantCount);

  std::error_code EC;
  llvm::raw_fd_ostream outputFile(mlir_name, EC);

  // Print the module header first
  outputFile << "\"builtin.module\"() <{sym_name = \"" << mlir_name
             << "\"}> ({\n";

  // Find and print the function declaration first
  mlir::func::FuncOp mainFunc;
  c->mod->walk([&](mlir::func::FuncOp funcOp) {
    if (!funcOp->hasAttr("is_comment_marker")) {
      mainFunc = funcOp;
    }
  });

  if (mainFunc) {
    // Print function signature with proper indentation
    outputFile << "  \"func.func\"() <{";

    // Print function type
    outputFile << "function_type = ";
    mainFunc.getFunctionType().print(outputFile);

    // Print symbol name
    outputFile << ", sym_name = \"" << mainFunc.getName() << "\"";

    // Print any other attributes the function might have
    for (auto attr : mainFunc->getAttrs()) {
      if (attr.getName() != "sym_name" &&
          attr.getName() != mlir::SymbolTable::getSymbolAttrName() &&
          attr.getName() != "function_type") {
        outputFile << ", " << attr.getName().str() << " = ";
        attr.getValue().print(outputFile);
      }
    }

    outputFile << "}> ({\n";

    // Print the function arguments block
    outputFile << "  ^bb0(";
    bool firstArg = true;
    for (auto arg : mainFunc.getArguments()) {
      if (!firstArg) {
        outputFile << ", ";
      }
      outputFile << "%arg" << arg.getArgNumber() << ": ";
      arg.getType().print(outputFile);

      // Check if location exists and print it
      outputFile << " ";
      mlir::Location loc = arg.getLoc();
      if (!loc.isa<mlir::UnknownLoc>()) {
        loc.print(outputFile);
      }
      firstArg = false;
    }
    outputFile << "):\n";
  }

  // Now print all the operations and comments in order
  int opNumber = 0;
  c->mod->walk([&](mlir::Operation *op) {
    // Skip the module and function ops since we already handled them
    if (mlir::isa<mlir::ModuleOp>(op) || op == mainFunc.getOperation()) {
      return;
    }

    if (op->hasAttr("is_comment_marker")) {
      auto commentAttr = op->getAttrOfType<mlir::StringAttr>("comment");
      if (commentAttr) {
        outputFile << commentAttr.getValue().str() << "\n";
      }
    } else if (mainFunc && op->getParentOp() == mainFunc.getOperation()) {
      // This is an operation inside the function body
      outputFile << "    ";

      // Print SSA value assignment for operations with results
      if (op->getNumResults() > 0) {
        outputFile << "%" << opNumber << " = ";
        opNumber++;
      }

      // Special handling for return operation to ensure correct format
      if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        outputFile << "\"func.return\"(";

        // Print operands
        bool firstOperand = true;
        for (auto operand : returnOp.getOperands()) {
          if (!firstOperand) {
            outputFile << ", ";
          }

          if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
            outputFile << "%arg" << blockArg.getArgNumber();
          } else {
            // Find the SSA value name by counting operations
            auto definingOp = operand.getDefiningOp();
            int operandOpNumber = 0;
            bool found = false;
            mainFunc.walk([&](mlir::Operation *innerOp) {
              if (innerOp == definingOp) {
                found = true;
                return;
              }
              if (!found && innerOp != mainFunc &&
                  !innerOp->hasAttr("is_comment_marker") &&
                  innerOp->getNumResults() > 0) {
                operandOpNumber++;
              }
            });
            outputFile << "%" << operandOpNumber;
          }

          firstOperand = false;
        }

        // Print types with correct format
        outputFile << ") : (";
        firstOperand = true;
        for (auto operand : returnOp.getOperands()) {
          if (!firstOperand) {
            outputFile << ", ";
          }
          operand.getType().print(outputFile);
          firstOperand = false;
        }

        // Add arrow syntax and empty return type
        outputFile << ") -> () ";

        // Add location if available
        if (!op->getLoc().isa<mlir::UnknownLoc>()) {
          op->getLoc().print(outputFile);
        } else {
          outputFile << "unknown";
        }

      } else {
        // Regular operation printing
        // Create a custom printer that excludes the data attribute and location
        mlir::OpPrintingFlags flags;
        flags.skipRegions();
        flags.elideLargeElementsAttrs(0);
        flags.printGenericOpForm();
        flags.enableDebugInfo(false);

        // Print operation name and types
        outputFile << "\"" << op->getName() << "\"(";

        // Print operands
        bool firstOperand = true;
        for (auto operand : op->getOperands()) {
          if (!firstOperand) {
            outputFile << ", ";
          }

          if (auto blockArg = operand.dyn_cast<mlir::BlockArgument>()) {
            outputFile << "%arg" << blockArg.getArgNumber();
          } else {
            // Find the SSA value name by counting operations
            auto definingOp = operand.getDefiningOp();
            int operandOpNumber = 0;
            bool found = false;
            mainFunc.walk([&](mlir::Operation *innerOp) {
              if (innerOp == definingOp) {
                found = true;
                return;
              }
              if (!found && innerOp != mainFunc &&
                  !innerOp->hasAttr("is_comment_marker") &&
                  innerOp->getNumResults() > 0) {
                operandOpNumber++;
              }
            });
            outputFile << "%" << operandOpNumber;
          }

          firstOperand = false;
        }

        // Print attributes except data
        bool hasAttrs = false;
        for (auto attr : op->getAttrs()) {
          if (attr.getName() != "data") {
            if (!hasAttrs) {
              outputFile << ") <{";
              hasAttrs = true;
            } else {
              outputFile << ", ";
            }
            outputFile << attr.getName().str() << " = ";
            attr.getValue().print(outputFile);
          }
        }

        if (hasAttrs) {
          outputFile << "}>";
        } else {
          outputFile << ")";
        }

        // Print types
        outputFile << " : (";
        firstOperand = true;
        for (auto operand : op->getOperands()) {
          if (!firstOperand) {
            outputFile << ", ";
          }
          operand.getType().print(outputFile);
          firstOperand = false;
        }
        outputFile << ") -> (";
        firstOperand = true;
        for (auto result : op->getResults()) {
          if (!firstOperand) {
            outputFile << ", ";
          }
          result.getType().print(outputFile);
          firstOperand = false;
        }
        outputFile << ")";

        // Single newline for operation
        outputFile << "\n";
      }
    }
  });

  // Close the function and module with the correct format
  if (mainFunc) {
    outputFile << "  }) : () -> () ";

    // Add location if available, otherwise use "unknown"
    if (!mainFunc->getLoc().isa<mlir::UnknownLoc>()) {
      mainFunc->getLoc().print(outputFile);
    } else {
      outputFile << "unknown";
    }

    outputFile << "\n";
  }

  // Close the module
  outputFile << "}) : () -> () loc(unknown)\n";

  c->valueMap.clear();
  c->funcMap.clear();
  c->mod.release();
  return mlir_name;
}

/* Unary Operation */
valueID createUnary(Context *ctxt, size_t opcode, valueID lhs, char *filename,
                    size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);
  auto &&source = valueMap[lhs];
  switch (opcode) {
  case 0: {
    auto res = builder.create<hecate::earth::BootstrapOp>(location, source);
    valueMap.push_back(res);
    break;
  }
  case 13: {
    auto res = builder.create<hecate::earth::NegateOp>(location, source);
    valueMap.push_back(res);
    break;
  }

  default:
    assert(0 && "Unary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

/* Binary Operation */
valueID createBinary(Context *ctxt, size_t opcode, valueID lhs, valueID rhs,
                     char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&valueMap = ctxt->valueMap;
  auto location =
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0);

  auto &&srcl = valueMap[lhs];
  auto &&srcr = valueMap[rhs];

  switch (opcode) {
  case 6: {
    auto res = builder.create<hecate::earth::AddOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }
  case 7: {
    auto neg = builder.create<hecate::earth::NegateOp>(location, srcr);
    valueMap.push_back(neg);
    auto res = builder.create<hecate::earth::AddOp>(location, srcl, neg);
    valueMap.push_back(res);
    break;
  }

  case 8: {
    auto res = builder.create<hecate::earth::MulOp>(location, srcl, srcr);
    valueMap.push_back(res);
    break;
  }

  default:
    assert(0 && "Binary Operation type is wrong");
  }
  return valueMap.size() - 1;
}

valueID createRotation(Context *ctxt, size_t valueID, int offset,
                       char *filename, size_t line) {
  auto &&builder = *ctxt->builder;
  auto &&srcl = ctxt->valueMap[valueID];
  auto cons = builder.create<hecate::earth::RotateOp>(
      mlir::FileLineColLoc::get(builder.getStringAttr(filename), line, 0), srcl,
      offset);
  ctxt->valueMap.push_back(cons);
  return ctxt->valueMap.size() - 1;
}

void setOutput(Context *ctxt, funcID fun, valueID *ret, size_t len) {
  llvm::SmallVector<mlir::Value, 2> rets;
  llvm::SmallVector<mlir::Type, 1> types;
  for (int i = 0; i < len; i++) {
    rets.push_back(ctxt->valueMap[ret[i]]);
    types.push_back(ctxt->valueMap[ret[i]].getType());
  }
  auto func = ctxt->funcMap[fun];
  ctxt->builder->create<mlir::func::ReturnOp>(func.getLoc(), rets);
  auto retType = func.getFunctionType();
  func.setFunctionType(
      ctxt->builder->getFunctionType(retType.getInputs(), types));
}

Context *init() {
  /* signal(SIGSEGV, handler); */
  return new ::hecate::Context();
}
void finalize(Context *ctxt) { delete ctxt; }

void addComment(Context *ctxt, char *comment_text, char *filename,
                size_t line) {
  std::string comment(comment_text);

  ctxt->pendingComments.push_back(comment);

  auto location = mlir::FileLineColLoc::get(
      ctxt->builder->getStringAttr(filename), line, 0);

  auto commentAttr =
      mlir::StringAttr::get(ctxt->builder->getContext(), comment);
  auto markerOp = ctxt->builder->create<mlir::func::FuncOp>(
      location, "_comment_" + std::to_string(ctxt->commentCount++),
      mlir::FunctionType::get(ctxt->builder->getContext(), {}, {}));

  markerOp->setAttr(
      "sym_visibility",
      mlir::StringAttr::get(ctxt->builder->getContext(), "private"));

  markerOp->setAttr("comment", commentAttr);
  markerOp->setAttr("is_comment_marker",
                    mlir::UnitAttr::get(ctxt->builder->getContext()));
}

// New functions to access value map and constant data
size_t getValueMapSize(Context *ctxt) { return ctxt->valueMap.size(); }

// Check if value is a constant
char *isConstant(Context *ctxt, valueID id) {
  if (id >= ctxt->valueMap.size()) {
    return nullptr;
  }

  auto value = ctxt->valueMap[id];

  if (auto constOp = value.getDefiningOp<hecate::earth::ConstantOp>()) {
    return "Constant";
  }
  return nullptr;
}

// Get constant values as array (for constants only)
double *getConstantValues(Context *ctxt, valueID id, size_t *outSize) {
  if (id >= ctxt->valueMap.size()) {
    *outSize = 0;
    return nullptr;
  }

  auto value = ctxt->valueMap[id];
  auto constOp = value.getDefiningOp<hecate::earth::ConstantOp>();
  if (!constOp) {
    *outSize = 0;
    return nullptr;
  }

  // Try to get the array data
  if (auto arrayAttr = constOp->getAttrOfType<mlir::ArrayAttr>("data")) {
    *outSize = arrayAttr.size();
    double *result = (double *)malloc(sizeof(double) * (*outSize));

    for (size_t i = 0; i < *outSize; i++) {
      if (auto floatAttr = arrayAttr[i].dyn_cast<mlir::FloatAttr>()) {
        result[i] = floatAttr.getValueAsDouble();
      } else {
        result[i] = 0.0; // fallback
      }
    }
    return result;
  }

  *outSize = 0;
  return nullptr;
}
} // extern "C"
} // namespace hecate

/* int main() {} */