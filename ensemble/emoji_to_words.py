import emoji

def emoji_to_words(emoji_original):
	emoji_value = emoji.demojize(emoji_original)
	emoji_list = emoji_value.split(':')
	emoji_string = ' '.join(emoji_list)
	emoji_string = emoji_string.replace('_',' ')
	emoji_string = emoji_string.replace('  ',' ')
	if emoji_string[0] == ' ':
		emoji_string = emoji_string[1:]
	return emoji_string
