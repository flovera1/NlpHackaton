def cleaning(text):
  """
  This function gets rid of the commas and \\n signs.
  Input should be a list of strings (phrases)
  """
  txt = []
  for sentence in text:
    sen = ''
    for string in sentence:
      string = string.replace(",","")
      string = string.replace("\n","")
      sen += string
    txt += [sen]
  print(txt)
