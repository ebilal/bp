# bp
Numpy implementation of the backpropagation algorithm with momentum and weight decay. It implements dense layers as objects of class Dense that can be strung together via a python list into a deep neural network.

How to run:

```python
from bp import Dense, predict, train

l1 = Dense(X_train.shape[1], 2000, activation='relu')
l2 = Dense(l1.output_dim, 200, activation='relu')
l3 = Dense(l2.output_dim, 200, activation='relu')
l4 = Dense(l3.output_dim, 200, activation='relu')
l5 = Dense(l4.output_dim, 200, activation='relu')
l6 = Dense(l3.output_dim, 2000, activation='relu')
l7 = Dense(l1.output_dim, 10, activation='softmax')

net = [l1, l2, l3, l4, l5, l6, l7]
train(net, X_train, Y_train, batch_size=100, epochs=10, lr=0.01, momentum=0., decay=0., verbose=100)
(pred_train, loss, acc) = predict(net, X_train, Y_train)
print('Train acc: {} - loss: {}'.format(acc, loss))
(pred_test, loss, acc) = predict(net, X_test, Y_test)
print('Test acc: {} - loss: {}'.format(acc, loss))
```

The ```decay``` variable
