import numpy as np

def unpack_matrix(m, num_of_rows, num_of_columns):
	# sanity checks
    n_rows, n_cols = m.shape
    if (n_rows % num_of_rows != 0 or n_cols % num_of_columns != 0):
        print(f'cannot unpack {m.shape} with dimensions ({num_of_rows}, {num_of_columns}) without losing information')
        
	# calculate sub-matrices sizes
    sub_matrix_width = n_cols // num_of_columns
    sub_matrix_height = n_rows // num_of_rows

    unpacked = {}    
    for i in range(n_rows // sub_matrix_height):
        for j in range(n_cols // sub_matrix_width):
            unpacked[(i, j)] = m[sub_matrix_height*i:sub_matrix_height*(i+1), sub_matrix_width*j:sub_matrix_width*(j+1)]
            
    return unpacked

def pack_matrix(upm, num_of_rows, num_of_cols):
    # reconstruct original shape
    sub_matrix_width, sub_matrix_height = upm[(0, 0)].shape
    full_matrix_width, full_matrix_height = sub_matrix_width*num_of_rows, sub_matrix_height*num_of_cols
    full_matrix = np.zeros((full_matrix_width, full_matrix_height))
    
    # copy submatrices inside the full matrix at the right locations 
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            full_matrix[sub_matrix_height*i:sub_matrix_height*(i+1), sub_matrix_width*j:sub_matrix_width*(j+1)] = upm[(j, i)]
    
    return full_matrix
    

## generate a small matrix
#a = np.random.randint(0, 10, size=(10, 10))
#print(f'Starting matrix=\n{a}')
#
## unpack and repack 
#upm = unpack_matrix(a, 2, 2)
#pa = pack_matrix(upm, 2, 2)
#
## show the (0, 0) submatrix 
#print(f'The (0, 0) submatrix:\n{upm[(0, 0)]}')
#
## print the recontructed matrix 
#print(f'Final matrix=\n{np.asarray(pa, dtype=np.int)}')
