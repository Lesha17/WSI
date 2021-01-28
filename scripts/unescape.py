import sys
from html import unescape

if __name__ == '__main__':
    in_filepath = sys.argv[1]
    out_filepath = sys.argv[2]
    with open(in_filepath) as in_f:
        with open(out_filepath, 'w') as out_f:
            for line in in_f:
                out_f.write(unescape(line))

