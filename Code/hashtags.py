def main():
    hashtag = "#FuckTrumpManMAN"
    hashtag_string = split_hashtags(hashtag)
    print(hashtag_string)

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
        if len(newword) > 0:
            hashtag_string = " ".join(newwordlist + [newword])
        else:
            hashtag_string = " ".join(newwordlist)        
    if lower:
        hashtag_string = hashtag_string[0].lower() + hashtag_string[1:]
    return hashtag_string

if __name__ == '__main__':
    main()