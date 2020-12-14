import stanza

stanza.download('en')   
nlp = stanza.Pipeline('en')

doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')


root_words = []
child_words =[]

def root_generator():
    for sent in doc.sentences:
        for word in sent.words:
                if(word.head == 0):
                    {
                        root_words.append(word.text)
                    }
                return root_words

def child_generator():
    for sent in doc.sentences:
        for word in sent.words:
                if(word.head > 0):
                    {
                        child_words.append(word.text)
                    }
                return child_words

