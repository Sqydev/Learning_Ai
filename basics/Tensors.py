import torch
import numpy as np

# What's Tensor NumpyArray and Array
    # Array in python is array from c but able to hold multiple types but super slow and can be only one dimencional(no matracies) and when summing it merges array's insted of suming them
    # NumpyArray is Python Array but able to hold data of only one type and can be multidimetynal(can do matracies) so faster
    # And Tensor is basicly Numbpy array but can work on CUDA nad it doesen't merge arrray's when adding two up but acualy sum's it. And overall is super fast

# Create data sample
data_array = [[1, 2], [3, 4]]

# Create numpy array from data sample
numpy_array = np.array(data_array)

# You can create tensor directry from data sample(f.e. data_array)
tensor_from_data = torch.tensor(data_array)

# Print Tensor
print(f"Tensor from data: \n {tensor_from_data} \n")

# Or from numpy array
tensor_from_numpy_array = torch.from_numpy(numpy_array)

# Print Tensor
print(f"Tensor from NumpyArray: \n {tensor_from_numpy_array} \n")


# Create new Tensor that is basicly identical to source Tensor but with all the data set to 1. So src_tensor = ([3, 2], [4, 1]) -> new_tensor = ([1, 1], [1, 1]) BUT you can also chainge aspects so you can
# chainge from int to float like that tensor_with_all_data_as_one = torch.ones_like(tensor_from_data, dtype=torch.float) here's rest of things you can chainge: 
# https://docs.pytorch.org/docs/stable/generated/torch.ones_like.html#torch.ones_like
tensor_with_all_data_as_one = torch.ones_like(tensor_from_data)

# And there's also zeros_like() that is litteraly ones_like() but with zeros

# Print Tensor
print(f"Ones Tensor: \n {tensor_with_all_data_as_one} \n")

# Same as tensor_with_all_data_as_one but all data is random and if you want you can overwrite the data type(but if you want int type(so long, int, itp) rand_like you have to make it randint_like
# and in randint_like(here) you have to add spectrum for randomization in the middle so randint_like(tensor_from_data, 0, 10, dtype=torch.int) will do same thing as rand_like(tensor_from_data, dtype=torch.float)
# but insted of it randomizing numbers from 0 to 1 it will randomise numbers from 0 to 10(and you can also give spectrum to float like rands but that will be later if it's usefull))
tensor_with_random_data = torch.rand_like(tensor_from_data, dtype=torch.float)

# Print Tensor
print(f"Random Tensor: \n {tensor_with_random_data} \n")

# You can also create tensor from shape like so:
shape = (3, 2)

shape_tensor = torch.ones(shape)
print(f"Tensor from shape: \n {shape_tensor} \n")

# Note: when you're making tensor from shape use torch.ones/zeros/rand itp. and NOT torch.ones_like/zeros_like/rand_like!!!!!!!!!

# And you can also print idividual atrybutes of an tensor
tensor_attributes_show = torch.tensor([[2, 5], [2, 5]])

print(f"Shape of tensor_attributes_show: {tensor_attributes_show.shape}")
print(f"Datatype of tensor_attributes_show: {tensor_attributes_show.dtype}")
print(f"Device tensor_attributes_show is stored on: {tensor_attributes_show.device} \n")

# From now on I'll use only tensor tensor
tensor = torch.tensor([[1, 4, 2], [4, 6, 1]])

# Also. you can do tensor operations like moving tensor to gpu
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print(f"Device tensor is stored on: {tensor.device} \n")

# SLICING:)
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(f"Fun slicing: {tensor} \n")

# Also you can join, multyply, matrix_mul, in place operations, yare yare

# Also tensor to numpy array:
ten_to_numpy_array = tensor.numpy()
print(f"Numpy arr from tensor: {ten_to_numpy_array} \n")

# Also, numpy array from tensor is something like pointer cuz if src tensor chainges numpy array chainges too
tensor.add_(1)
print(f"src tensor after addition: {tensor} \n")
print(f"numpy array without adding but made from tensor that is after addition: {ten_to_numpy_array} \n")

# Btw it is the same with tensor from numpy array but i'm to lazy to do example for this
