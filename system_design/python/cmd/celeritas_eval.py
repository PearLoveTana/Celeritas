import sys

import celeritas as m


def main():
    m.celeritas_eval(len(sys.argv), sys.argv)


if __name__ == "__main__":
    sys.exit(main())
