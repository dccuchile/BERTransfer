from cleantext import clean
import sys

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage: python clean_bookcorpus.py <bookcorpus_file> <processed_text_file>')
		sys.exit(1)
	in_f = sys.argv[1]
	out_f = sys.argv[2]
	in_f = open(in_f,"r")
	output = open(out_f, 'w')
#	output.write(" ")
	for line in in_f:
		cleaned_text = clean(line,
		    fix_unicode=True,               # fix various unicode errors
		    to_ascii=True,                  # transliterate to closest ASCII representation
		    lower=True,                     # lowercase text
		    no_line_breaks=False,           # fully strip line breaks as opposed to only normalizing them
		    no_urls=True,                  # replace all URLs with a special token
		    no_emails=True,                # replace all email addresses with a special token
		    no_phone_numbers=False,         # replace all phone numbers with a special token
		    no_numbers=True,               # replace all numbers with a special token
		    no_digits=False,                # replace all digits with a special token
		    no_currency_symbols=True,      # replace all currency symbols with a special token
		    no_punct=True,                 # fully remove punctuation
		    replace_with_url="<URL>",
		    replace_with_email="<EMAIL>",
		    replace_with_phone_number="<PHONE>",
		    replace_with_number="<NUM>",
		    replace_with_digit="<DIGIT>",
		    replace_with_currency_symbol="<CUR>",
		    lang="en"                       # set to 'de' for German special handling
		)
		output.write(cleaned_text+" ") # lines
	in_f.close()
	output.close()
