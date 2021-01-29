import pickle as pkl

path = r"C:\Users\piotr\Desktop\Piotr\studia-praca\Part_II_project\Music-Generation\run\two_datasets_attention\store\distincts"

with open(path, 'rb') as f:
    data = pkl.load(f)

print(data)