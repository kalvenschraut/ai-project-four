import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) < 0:
            return -1
        return 1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        failed = True
        while failed == True:
            failed = False
            for x, y in dataset.iterate_once(1):
                y_scalar = nn.as_scalar(y);
                if self.get_prediction(x) != y_scalar:
                    self.get_weights().update(x, y_scalar)
                    failed = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.hidden_layers = 1
        self.hidden_layer_size = 512
        self.batch_size = 200
        self.learning_rate = -0.05
        self.output_size = 1
        self.m = []
        self.b = []

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        if len(self.m) == 0:
            n = x.data.size
            m = self.hidden_layer_size
            for layers in range(self.hidden_layers + 1):
                if layers == self.hidden_layers:
                    n = m
                    m = self.output_size
                elif layers > 0:
                    temp_m = m
                    m = n
                    n = temp_m
                self.m.append(nn.Parameter(n, m))
                self.b.append(nn.Parameter(1, m))
        predicted_y = x;
        for i, m in enumerate(self.m):
            b = self.b[i]
            xm = nn.Linear(predicted_y, m)
            predicted_y = nn.AddBias(xm, b)
            if i != len(self.m) - 1:
                predicted_y = nn.ReLU(predicted_y)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            loss = self.get_loss(x, y)
            if nn.as_scalar(loss) < 0.01:
                break
            parameters = []
            for i, m in enumerate(self.m):
                b = self.b[i]
                parameters.append(m)
                parameters.append(b)
            gradients = nn.gradients(loss, parameters)
            for i, m in enumerate(self.m):
                m.update(gradients[i * 2], self.learning_rate)
                self.b[i].update(gradients[(i * 2) + 1], self.learning_rate)


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).
(See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.m = []
        self.b = []
        # Initialize your model parameters here
        self.hidden_layers = 1
        self.hidden_layer_size = 100
        self.batch_size = 100
        self.learning_rate = -0.5
        self.input_size = 784
        self.output_size = 10
        self.m = []
        self.b = []

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        if len(self.m) == 0:
            n = x.data.size
            m = self.hidden_layer_size
            for layers in range(self.hidden_layers + 1):
                if layers == self.hidden_layers:
                    n = m
                    m = self.output_size
                elif layers > 0:
                    temp_m = m
                    m = n
                    n = temp_m
                self.m.append(nn.Parameter(n, m))
                self.b.append(nn.Parameter(1, m))
        predicted_y = x;
        for i, m in enumerate(self.m):
            b = self.b[i]
            xm = nn.Linear(predicted_y, m)
            predicted_y = nn.AddBias(xm, b)
            if i != len(self.m) - 1:
                predicted_y = nn.ReLU(predicted_y)
        return predicted_y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            if dataset.get_validation_accuracy() > .975:
                break
            loss = self.get_loss(x, y)
            parameters = []
            for i, m in enumerate(self.m):
                b = self.b[i]
                parameters.append(m)
                parameters.append(b)
            gradients = nn.gradients(loss, parameters)
            for i, m in enumerate(self.m):
                m.update(gradients[i * 2], self.learning_rate)
                self.b[i].update(gradients[(i * 2) + 1], self.learning_rate)

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        self.hidden_layer_size = 100
        self.batch_size = 100
        self.output_size = 5;
        self.learning_rate = -0.15
        self.W1 = nn.Parameter(self.num_chars,self.hidden_layer_size)
        self.b1 = nn.Parameter(1,self.hidden_layer_size)
        self.W2 = nn.Parameter(self.batch_size,self.hidden_layer_size)
        self.b2 = nn.Parameter(1,self.hidden_layer_size)
        self.W1_hidden = nn.Parameter(self.batch_size,self.hidden_layer_size)
        self.b1_hidden = nn.Parameter(1,self.hidden_layer_size)
        self.W2_hidden = nn.Parameter(self.batch_size,self.hidden_layer_size)
        self.b2_hidden = nn.Parameter(1,self.hidden_layer_size)
        self.W_end = nn.Parameter(self.hidden_layer_size,self.output_size)
        self.b_end = nn.Parameter(1,5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(xs)):
            if i==0:
                Z1 = nn.AddBias(nn.Linear(xs[i], self.W1), self.b1)
                A1 = nn.ReLU(Z1);
                h = nn.AddBias(nn.Linear(A1, self.W2), self.b2)
            else:
                Zi = nn.AddBias(nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(h, self.W1_hidden)), self.b1_hidden)
                Ai = nn.ReLU(Zi)
                Z_next = nn.AddBias(nn.Linear(Ai, self.W2_hidden), self.b2_hidden)
                h = nn.ReLU(Z_next)
        return nn.AddBias(nn.Linear(h, self.W_end), self.b_end)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        ans = self.run(xs)
        return nn.SoftmaxLoss(ans,y)

    def train(self, dataset):
        """
        Trains the model.
        """
        for x, y in dataset.iterate_forever(self.batch_size):
            if dataset.get_validation_accuracy() > .87:
                break
            grad_wrt_W1, grad_wrt_b1, grad_wrt_W2, grad_wrt_b2, grad_wrt_W1_hidden, grad_wrt_b1_hidden, grad_wrt_W2_hidden, grad_wrt_b2_hidden, grad_wrt_W_end, grad_wrt_b_end = nn.gradients(self.get_loss(x,y), [self.W1, self.b1, self.W2, self.b2, self.W1_hidden, self.b1_hidden, self.W2_hidden, self.b2_hidden, self.W_end, self.b_end])
            self.W1.update(grad_wrt_W1, self.learning_rate)
            self.b1.update(grad_wrt_b1, self.learning_rate)
            self.W2.update(grad_wrt_W2, self.learning_rate)
            self.b2.update(grad_wrt_b2, self.learning_rate)
            self.W1_hidden.update(grad_wrt_W1_hidden, self.learning_rate)
            self.b1_hidden.update(grad_wrt_b1_hidden, self.learning_rate)
            self.W2_hidden.update(grad_wrt_W2_hidden, self.learning_rate)
            self.b2_hidden.update(grad_wrt_b2_hidden, self.learning_rate)
            self.W_end.update(grad_wrt_W_end, self.learning_rate)
            self.b_end.update(grad_wrt_b_end, self.learning_rate)
