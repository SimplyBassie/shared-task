def main():
    file = open("../Data/not_offensive_words_scrape.txt", "r")
    f = file.read().strip()
    f.replace("-",",")
    f.replace("\n", ",")
    whitelist = f.split(",")
    print(whitelist)
    file.close()
    f2=open("../Data/not_offensive_words.txt", "a+")
    for word in whitelist:
        word = word.lower().strip()
        print(word)
        f2.write(word + "\n")

if __name__ == '__main__':
    main()
