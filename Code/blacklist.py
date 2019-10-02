def blacklist_reader():
    file = open("../Data/offensive_words.txt", "r")
    f = file.read().strip()
    blacklist = f.split("\n")
    return blacklist

def main():
    blacklist = blacklist_reader()
    print(blacklist)

if __name__ == '__main__':
    main()
