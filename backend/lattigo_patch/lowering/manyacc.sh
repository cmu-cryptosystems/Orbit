#!/bin/bash
# python -u run_acc_many.py --model ResNet --act ReLU --n 64
# python -u run_acc_many.py --model ResNet --act SiLU --n 64
# python -u run_acc_many.py --model AlexNet --act ReLU --n 64
# python -u run_acc_many.py --model AlexNet --act SiLU --n 64
# python -u run_acc_many.py --model MobileNet --act ReLU --n 64
# python -u run_acc_many.py --model MobileNet --act SiLU --n 64
# python -u run_acc_many.py --model SqueezeNet --act ReLU --n 64
# python -u run_acc_many.py --model SqueezeNet --act SiLU --n 64
# python -u run_acc_many.py --model VGG16 --act ReLU --n 64
# python -u run_acc_many.py --model VGG16 --act SiLU --n 64
# echo "Check ResNet ReLU Plaintext Accuracy:"
# python -u check_one_test.py --model ResNet --act ReLU --n 64 --run 1000
# echo "Check ResNet SiLU Plaintext Accuracy:"
# python -u check_one_test.py --model ResNet --act SiLU --n 64 --run 1000
echo "Check AlexNet ReLU Plaintext Accuracy:"
python -u check_one_test.py --model AlexNet --act ReLU --n 64 --run 1000
echo "Check AlexNet SiLU Plaintext Accuracy:"
python -u check_one_test.py --model AlexNet --act SiLU --n 64 --run 1000
echo "Check MobileNet ReLU Plaintext Accuracy:"
python -u check_one_test.py --model MobileNet --act ReLU --n 64 --run 1000
echo "Check MobileNet SiLU Plaintext Accuracy:"
python -u check_one_test.py --model MobileNet --act SiLU --n 64 --run 1000
echo "Check SqueezeNet ReLU Plaintext Accuracy:"
python -u check_one_test.py --model SqueezeNet --act ReLU --n 64 --run 1000
echo "Check SqueezeNet SiLU Plaintext Accuracy:"
python -u check_one_test.py --model SqueezeNet --act SiLU --n 64 --run 1000
echo "Check VGG16 ReLU Plaintext Accuracy:"
python -u check_one_test.py --model VGG16 --act ReLU --n 64 --run 1000
echo "Check VGG16 SiLU Plaintext Accuracy:"
python -u check_one_test.py --model VGG16 --act SiLU --n 64 --run 1000
echo "All Done."