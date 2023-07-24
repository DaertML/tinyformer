def one_hot_encode(number, max_length):
  one_hot_list = []
  for i in range(max_length):
    if i == number:
      one_hot_list.append(1)
    else:
      one_hot_list.append(0)
  return one_hot_list

def dot_product(A, B):
  result = []
  for i in range(len(A)):
    row = []
    for j in range(len(B[0])):
      dot = 0
      for k in range(len(A[0])):
        dot += A[i][k] * B[k][j]
      row.append(dot)
    result.append(row)
  return result

def transpose(A):
  result = []
  for j in range(len(A[0])):
    col = []
    for i in range(len(A)):
      col.append(A[i][j])
    result.append(col)
  return result

import math

def positional_encoding(max_len, d_model):
  pos_enc = [[0] * d_model for _ in range(max_len)]
  for pos in range(max_len):
    for i in range(0, d_model, 2):
      pos_enc[pos][i] = math.sin(pos / (10000 ** (i / d_model)))
      pos_enc[pos][i + 1] = math.cos(pos / (10000 ** (i / d_model)))
  return pos_enc

def scalar_product(lst, scalar):
  result = []
  for sublist in lst:
    new_sublist = []
    for element in sublist:
      new_sublist.append(element * scalar)
    result.append(new_sublist)
  return result

def concat_by_columns(list1, list2):
    output = []
    for row1, row2 in zip(list1, list2):
        new_row = row1 + row2
        output.append(new_row)
    return output

def mean_and_var(list_of_lists):
  all_elements = []
  for sublist in list_of_lists:
    all_elements.extend(sublist)
  mean = sum(all_elements) / len(all_elements)
  variance = sum((x - mean) ** 2 for x in all_elements) / len(all_elements)
  return (mean, variance)

def scalar_sum_vct(lst, value):
  return [x + value for x in lst]

def scalar_sum_mat(lst_of_lsts, value):
  return [[x + value for x in sublist] for sublist in lst_of_lsts]

def relu(list_of_lists):
  output = []
  for sublist in list_of_lists:
    new_sublist = []
    for element in sublist:
      new_element = max(0, element)
      new_sublist.append(new_element)
    output.append(new_sublist)
  return output

def generate_mask_matrix(n):
  mask = []
  for i in range(n):
    row = []
    for j in range(n):
      if j <= i:
        row.append(1)
      else:
        row.append(-10000)
    mask.append(row)
  return mask

def sum_mat(list1,list2):
    result = []
    for sublist1, sublist2 in zip(list1, list2):
        sublist = []
        for num1, num2 in zip(sublist1, sublist2):
            sublist.append(num1 + num2)
        result.append(sublist)

    return result