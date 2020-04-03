from code import Decoder

encoded_text = open('encoded_text.txt').read()
print(Decoder(encoded_text).decode())
