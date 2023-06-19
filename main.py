import argparse

if __name__ == '__main__':
    print('PyCharm')
    parser = argparse.ArgumentParser(description='ex')

    parser.add_argument('-p1', '--parser1', required=False, type=str, default='p1', help='the first parser')
    parser.add_argument('-p2', type=int, default=2222222, help='the second parser')
    args = parser.parse_args()
    print(args.parser1)
    print(args.p2)

    print('Argments in experiment:\n', args)

    setting = '{}_{}'.format(args.parser1, args.p2)
    for i in range(args.itr):
        print(i)
