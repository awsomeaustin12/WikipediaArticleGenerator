#dependencie

import numpy as np #vectorization
import random #generate probability distribution
import tensorflow as tf
import datetime #clock training time




text = open('wiki.test.tokens',encoding="utf8").read()
text = text[:50000]
print('text length in number of characters', len(text))

print('head of text:')
print(text[:1000])

#preprocessing ste
#print ot or characters and sort them
chars = sorted(list(set(text)))     #took out duplicates to make a set, then made it a list, then sorted the godamn list
char_size = len(chars)
print('number of characters',char_size)
print(chars)

#these are all the characters we are going to use
#we need a method to convert characters to ids (this will help with mapping)
char2id  = dict((c,i) for i, c in enumerate(chars))
id2char = dict((i,c) for i, c in enumerate(chars))

#now make a helper method
#generates a probability of each next character
def sample(prediction):
    r = random.uniform(0,1)
    #store prediction character
    s = 0
    char_id = len(prediction)-1  #its -1 because the length is greater tan the indice starting at 0
    #for each char prediction probability
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break

    char_one_hot = np.zeros(shape=char_size)
    char_one_hot[char_id] = 1.0
    return char_one_hot


#now we have to vectorize our data
len_per_section = 50  #50 character batches per sentence
skip = 2
sections = []
next_chars = []

for i in range(0,len(text)-len_per_section,skip):
    sections.append(text[i: i+len_per_section])
    next_chars.append(text[i+len_per_section])

#this gives us the chunks of data we need
X = np.zeros((len(sections),len_per_section,char_size))
y = np.zeros((len(sections),char_size))

#for each char in each section, convert each char to an id
#for each section convert the labe to an id
for i, sections in enumerate(sections):
    for j, char in enumerate(sections):
        X[i,j,char2id[char]] = 1
    y[i,char2id[next_chars[i]]] = 1
print(y)

#now for the machine learning part
#addng some VR into the mix

batch_size =512   #number of example put into the training samples
     #one epoch is a a full forward and backward pass
max_steps = 70000   #num of ierations
log_every = 100  #wanna print every 100 steps
save_every = 6000 #wanna save our model so we dont lose where we are in time
hidden_nodes = 1024 #dont want to little then it underfits but when you have to many hidden nodes it starts overfitting
                    #there is a formula for the perfect numbe of hidden nodes
                    #specifically for supervised learning

starting_text = 'i am thinking that'
#save our model
checkpoint_directory = 'ckpt'  #flags to identigy where in the teaining proces we are

#create a checkpoint directory
if tf.gfile.Exists(checkpoint_directory):
    tf.gfile.DeleteRecursively(checkpoint_directory)
tf.gfile.MakeDirs(checkpoint_directory)

print('training data size:', len(X))
print('approximate steps per epoch:', int(len(X)/batch_size))



#build our model


#now we actually build the LSTM network
#create computation graph
graph = tf.Graph()
with graph.as_default():
    global_step = tf.Variable(0) #number of batches seen by the graph

    data = tf.placeholder(tf.float32,[batch_size,len_per_section,char_size])
    labels = tf.placeholder(tf.float32,[batch_size,char_size])

    #input gate, output gate, forget gate, internal state
    #these will be calculated/updated through time
    #they are calculated independently

    #input gate - weights for input, weights for output, then bias
    w_ii = tf.Variable(tf.truncated_normal([char_size, hidden_nodes],-0.1,0.1))
    w_io = tf.Variable(tf.truncated_normal([hidden_nodes,hidden_nodes],-0.1,0.1))
    b_i = tf.Variable(tf.zeros([1,hidden_nodes]))

    #forget gate
    w_fi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes],-0.1,0.1))
    w_fo = tf.Variable(tf.truncated_normal([hidden_nodes,hidden_nodes],-0.1,0.1))
    b_f = tf.Variable(tf.zeros([1,hidden_nodes]))

    #output gate
    w_oi = tf.Variable(tf.truncated_normal([char_size, hidden_nodes],-0.1,0.1))
    w_oo = tf.Variable(tf.truncated_normal([hidden_nodes,hidden_nodes],-0.1,0.1))
    b_o = tf.Variable(tf.zeros([1,hidden_nodes]))

    #memory cell - internal hidden state
    w_ci = tf.Variable(tf.truncated_normal([char_size, hidden_nodes],-0.1,0.1))
    w_co = tf.Variable(tf.truncated_normal([hidden_nodes,hidden_nodes],-0.1,0.1))
    b_c = tf.Variable(tf.zeros([1,hidden_nodes]))

    def lstm(i,o,state):

        #nooverlap till we get to this

        #(input + input_weights) + (output * weights for prev ourput) + bias
                # as a quick note they are summing them together to show how the data has changed over time
        input_gate = tf.sigmoid(tf.math_ops.matmul(i,w_ii) + tf.math_ops.matmul(o,w_io)+b_i)

        # (input + forget_weights) + (output * weights for prev ourput) + bias
        forget_gate = tf.sigmoid(tf.math_ops.matmul(i,w_fi)+tf.math_ops.matmul(o,w_fo)+b_f)

        #(input + output_weights) + (output * weights for prev ourput) + bias
        output_gate = tf.sigmoid(tf.math_ops.matmul(i,w_oi) + tf.math_ops.matmul(o,w_oo) + b_o)

        # (input + internal_state_weights) + (output * weights for prev ourput) + bias
        memory_cell = tf.sigmoid(tf.math_ops.matmul(i,w_ci) + tf.math_ops.matmul(o,w_co) + b_c)

        #now that w have the 4 states we will start combineing them all together
        # .. now multiply (forget gate * given state) + (input_gate * hidden_State)
                #we do this to forget what we have given
                #and remember from what we have learned
        statew = forget_gate * state + input_gate * memory_cell

        #thn squash that state with the tanh nonlinearity
        output = output_gate * tf.tanh(state)

        #now return that shit
        return output , statew;

#operation for LSTM
#both output and state start off as empty
#the LSTM will calculate it on its inital run so you dont have to give it value intiailly

output = tf.zeros([batch_size,hidden_nodes])
state = tf.zeros([batch_size,hidden_nodes])

#unrolled LSTM loop
#for each input set
for i in range(len_per_section):
    #calculate state an output from LSTM
    output,state = lstm(data[:,i,:],output,state)
    #to start
    if i==0:
        #store all initial outputs and labels
        outputs_all_i = output
        labels_all_i = data[:,i+1,:]
    #for each new set concat outputs and labels
    elif i != len_per_section -1:
        #concatenate (combine) al vectors along a dimension axis not multiply
        outputs_all_i = tf.concat([outputs_all_i,output],axis=0)
        labels_all_i = tf.concat([labels_all_i,data[:,i+1,:]],axis=0)
    else:
        #final store
        outputs_all_i = tf.concat([outputs_all_i,output],axis=0)
        labels_all_i = tf.concat([labels_all_i,labels],axis=0)


#classifier
#will only run after aved_output and saved_state were assigned


#calculate the weight and bias values for the network
w = tf.Variable(tf.truncated_normal([hidden_nodes,char_size],-0.1,0.1))
b = tf.Variable(tf.zeros([char_size]))

#logits simply means that the function operates o the unscaled output
#of earlier layers and that the relative scale to understand the units
#id linera. It means, in particular, the sum of the inputs may ot equal 1
#that the values are not probabilites (you might have an input of 5)
logits = tf.matmul(outputs_all_i,w) + b

#logits is the predicted outputs
#compare it with the actual labels we have

#we use cross entropy since it is multiclass classification
#then computes the mean of elements across dimensions of a tensor
#averal loss across all values
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,labels_all_i))

#optimizer
#minimize the loss with gradient descent
optimizer = tf.train.GradientDescentOptimizer(10).minimize(loss,global_step=global_step)

#now we have to train the model usng this optimizer
#initiialize the sesion with a graph
with tf.Session(graph=graph) as sess:
    #standardize the initialize(init) step
    tf.global_variables_initializer().run()
    offset = 0
    saver = tf.train.Saver() #used to save the model every so often

    #for each training step
    for step in range(max_steps):

        #starts off as 0
        offset = offset % len(X)

        #calculate batch data and labels to feed model iterativey
        if offset <= (len(X) - batch_size):
            #first part
            batch_data = X[offset:offset+batch_size]
            batch_labels = y[offset:offset+batch_size]
            offset += batch_size
        #do this until offset = batch size
        else:
            #last part
            to_add = batch_size - (len(X)-offset)
            batch_data = np.concatenate([X[offset:len(X)], X[0:to_add]])
            batch_labels = np.concatenate([y[offset:len(X)], y[0:to_add]])
            offset = to_add
        #optimizer
        #have fun with this parrt
        _,training_loss = sess.run([optimizer,loss], feed_dict={data:batch_data,labels:batch_labels})

        if step % 10 == 0:
            print('training loss at step %d: %.2f (%s)' % (step,training_loss,datetime.datetime))

            if step % save_every == 0:
                saver.save(sess,checkpoint_directory+'/model',global_step=step)




                ###########
                # Test
                ###########
test_data = tf.placeholder(tf.float32, shape=[1, char_size])
test_output = tf.Variable(tf.zeros([1, hidden_nodes]))
test_state = tf.Variable(tf.zeros([1, hidden_nodes]))

                # Reset at the beginning of each test
reset_test_state = tf.group(test_output.assign(tf.zeros([1, hidden_nodes])),
test_state.assign(tf.zeros([1, hidden_nodes])))

                # LSTM
test_output, test_state = lstm(test_data, test_output, test_state)
test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)




test_start = 'I plan to make the world a better place '

with tf.Session(graph=graph) as sess:
    # init graph, load model
    tf.global_variables_initializer().run()
    model = tf.train.latest_checkpoint(checkpoint_directory)
    saver = tf.train.Saver()
    saver.restore(sess, model)

    # set input variable to generate chars from
    reset_test_state.run()
    test_generated = test_start

    # for every char in the input sentennce
    for i in range(len(test_start) - 1):
        # initialize an empty char store
        test_X = np.zeros((1, char_size))
        # store it in id from
        test_X[0, char2id[test_start[i]]] = 1.
        # feed it to model, test_prediction is the output value
        _ = sess.run(test_prediction, feed_dict={test_data: test_X})

    # where we store encoded char predictions
    test_X = np.zeros((1, char_size))
    test_X[0, char2id[test_start[-1]]] = 1.

    # lets generate 500 characters
    for i in range(500):
        # get each prediction probability
        prediction = test_prediction.eval({test_data: test_X})[0]
        # one hot encode it
        next_char_one_hot = sample(prediction)
        # get the indices of the max values (highest probability)  and convert to char
        next_char = id2char[np.argmax(next_char_one_hot)]
        # add each char to the output text iteratively
        test_generated += next_char
        # update the
        test_X = next_char_one_hot.reshape((1, char_size))

    print(test_generated)