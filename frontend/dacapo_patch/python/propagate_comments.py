import sys

def propagate_block_comments(input_path, output_path):
    comment_prefix = '//'
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        current_block_comment = None
        for line in infile:
            stripped = line.strip()
            if stripped.startswith(comment_prefix) and not stripped[len(comment_prefix):].strip().startswith('"'):
                current_block_comment = stripped
                continue
            if current_block_comment and stripped and not stripped.startswith(comment_prefix):
                if not current_block_comment.startswith("loc("):
                    current_block_comment = f"loc(unknown) {current_block_comment}"
                if "loc" in line and current_block_comment.startswith("loc("):
                    current_block_comment = current_block_comment[13:]
                if comment_prefix in line:
                    code, side = line.split(comment_prefix, 1)
                    new_line = f"{code.rstrip()} {comment_prefix} {side.strip()} {current_block_comment}\n"
                else:
                    new_line = line.rstrip('\n') + f"  {current_block_comment}\n"
                outfile.write(new_line)
            else:
                outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python propagate_comments.py input.mlir output.mlir")
        sys.exit(1)
    propagate_block_comments(sys.argv[1], sys.argv[2])