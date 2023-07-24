from functions import *
from const import *

max_len = 10
d_model = 4

input = "hi how are you"
output_pred = "i am fine"
output = "<sos> <pad> <pad> <pad>"

input_words = input.split(" ")
output_words = output.split(" ")

special_tokens = ["<sos>", "<eos>", "<pad>"] # may interest <pad>
vocabulary = list(sorted(set(input_words+output_words)))
vocabulary = vocabulary + special_tokens

##############################################################################
#           ENCODER
##############################################################################

input_onehots = [one_hot_encode(vocabulary.index(w), len(vocabulary)) for w in input_words]
input_embeds = dot_product(input_onehots, embedding_matrix)

pos = positional_encoding(max_len, d_model)[:len(input_embeds)]
embed_pos = sum_mat(input_embeds, pos)

Z = []
V = []
for i, mat in enumerate(Wq):
    Wqi = Wq[i]
    Wki = Wk[i]
    Wvi = Wv[i]

    Qi = dot_product(embed_pos, Wqi)
    Ki = dot_product(embed_pos, Wki)
    Vi = dot_product(embed_pos, Wvi)

    _QKt = dot_product(Qi,transpose(Ki))
    _QKt_sqrt_dk = scalar_product(_QKt, 1/math.sqrt(d_model))
    soft_num = [[math.exp(number) for number in row] for row in _QKt_sqrt_dk]
    soft = [[number/sum(row) for number in row] for row in _QKt_sqrt_dk]

    Zi = dot_product(soft, Vi)

    V.append(Vi)
    Z.append(Zi)
    if len(Z) > 1:
        concat_by_columns(Z[0], Z[1])
    if len(V) > 1:
        concat_by_columns(V[0], V[1])

ZWo = dot_product(Z[0], Wo)

AddNorm_input = sum_mat(ZWo,embed_pos)

mean, var = mean_and_var(AddNorm_input)
norm = scalar_product(scalar_sum_mat(AddNorm_input,mean), 1/(math.sqrt(var)))

ffl1 = relu(sum_mat(dot_product(norm, ffw1),ffb1))
ffl2 = sum_mat(dot_product(ffl1, ffw2),ffb2)
ffnn_out = sum_mat(ffl2, norm)

mean, var = mean_and_var(ffnn_out)
normffl2 = scalar_product(scalar_sum_mat(ffnn_out,mean), 1/(math.sqrt(var)))
encoder_out = normffl2

##############################################################################
#           DECODER
##############################################################################
generated_words = len(output_words)-1
mask_mat = generate_mask_matrix(generated_words)

for j in range(generated_words):
    output_onehots = [one_hot_encode(vocabulary.index(w), len(vocabulary)) for w in output_words]
    output_embeds = dot_product(output_onehots, embedding_matrix)

    pos = positional_encoding(max_len, d_model)[:len(output_embeds)]
    output_embed_pos = sum_mat(output_embeds, pos)

    Z = []
    V = []
    ### MASKED MULTIHEAD ATTENTION
    for i, mat in enumerate(Wq):
        Wqi = Wq[i]
        Wki = Wk[i]
        Wvi = Wv[i]

        Qi = dot_product(output_embed_pos, Wqi)
        Ki = dot_product(output_embed_pos, Wki)
        Vi = dot_product(output_embed_pos, Wvi)

        _QKt = dot_product(Qi,transpose(Ki))
        _masked_QKt = sum_mat(_QKt, mask_mat)
        _QKt_sqrt_dk = scalar_product(_masked_QKt, 1/math.sqrt(d_model))
        soft_num = [[math.exp(number) for number in row] for row in _QKt_sqrt_dk]
        soft = [[number/sum(row) for number in row] for row in _QKt_sqrt_dk]

        Zi = dot_product(soft, Vi)
        Z.append(Zi)
        if len(Z) > 1:
            concat_by_columns(Z[0], Z[1])

    ZWo = dot_product(Z[0], Wo)

    # ADD & NORM OF MASKED
    AddNorm_input = sum_mat(ZWo, output_embeds)
    mean, var = mean_and_var(AddNorm_input)
    norm = scalar_product(scalar_sum_mat(AddNorm_input,mean), 1/(math.sqrt(var)))

    # MULTIHEAD
    Qhead = norm
    Khead = encoder_out
    Vhead = encoder_out

    for i, mat in enumerate(Wq):
        for i, mat in enumerate(Wq):
            Wqi = Wq[i]
            Wki = Wk[i]
            Wvi = Wv[i]

            Qi = dot_product(Qhead, Wqi)
            Ki = dot_product(Khead, Wki)
            Vi = dot_product(Vhead, Wvi)

            _QKt = dot_product(Qi,transpose(Ki))
            _masked_QKt = sum_mat(_QKt, mask_mat)
            _QKt_sqrt_dk = scalar_product(_masked_QKt, 1/math.sqrt(d_model))
            soft_num = [[math.exp(number) for number in row] for row in _QKt_sqrt_dk]
            soft = [[number/sum(row) for number in row] for row in _QKt_sqrt_dk]

            Zi = dot_product(soft, Vi)
            Z.append(Zi)
            if len(Z) > 1:
                concat_by_columns(Z[0], Z[1])

        ZWo = dot_product(Z[0], Wo)
    
    # ADD & NORM OF MASKED
    AddNorm_input = sum_mat(ZWo, output_embeds)
    mean, var = mean_and_var(AddNorm_input)
    norm = scalar_product(scalar_sum_mat(AddNorm_input,mean), 1/(math.sqrt(var)))

    ffl1 = relu(sum_mat(dot_product(norm, ffw1),ffb1))
    ffl2 = sum_mat(dot_product(ffl1, ffw2),ffb2)

    ffnn_out = sum_mat(ffl2, norm)
    mean, var = mean_and_var(ffnn_out)
    normffl2 = scalar_product(scalar_sum_mat(ffnn_out,mean), 1/(math.sqrt(var)))
    decoder_out = normffl2

    # LAST LINEAR
    linear_out = dot_product(linear_w,transpose(decoder_out))
    linear_out = [[item for sublist in linear_out for item in sublist]]

    # LAST SOFTMAX
    soft_num = [[math.exp(number) for number in row] for row in linear_out]
    soft = [[number/sum(row) for number in row] for row in linear_out]
    max_soft = max(soft[0])
    next_word = vocabulary[soft[0].index(max_soft)]
    output_words[j+1] = next_word
    print(output_words)
    print(next_word)