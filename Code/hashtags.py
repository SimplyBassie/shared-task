def main():
    hashtag = "#fuckTrump"
    wordlist = split_hashtags(hashtag)
    print(wordlist)

def split_hashtags(hashtag):
    uppercount = 0
    lower = False
    hashtag = hashtag[1:]
    if hashtag[0].islower():
        lower = True
    hashtag = hashtag[0].upper() + hashtag[1:]
    wordlist = []
    ilist = []
    newword = ""
    for i in range(len(hashtag)):
        if hashtag[i].isupper():
            uppercount += 1
            ilist.append(i)
    if len(ilist) == 1:
        wordlist.append(hashtag[ilist[0]:])
    else:
        for i in range(len(ilist)-1):
            wordlist.append(hashtag[ilist[0]:ilist[1]])
            ilist.remove(ilist[0])
            if len(ilist) == 1:
                wordlist.append(hashtag[ilist[0]:])
    newwordlist = []
    for word in wordlist:
        if len(word) > 1 or word == "I":
            newwordlist.append(word)
        else:
            newword += word
    if lower and uppercount < 2:
        newwordlist = [newwordlist[0][0].lower()+newwordlist[0][1:]]
    if len(newword) > 0:
        return newwordlist + [newword]
    else:
        return newwordlist

if __name__ == '__main__':
    main()