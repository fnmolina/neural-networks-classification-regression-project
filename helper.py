

callback = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.001,
    patience=10,
    verbose=2,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=30,
)

K.clear_session()
def three_layer_model(lr=0.001, decay=0, hidden_units_1=100, hidden_units_2=100, activation='relu', droupout1 = 0.25, droupout2 = 0.25):
    output_size = 10
    model = Sequential()
    model.add(Flatten(input_shape=x_train.shape[1:]))
    model.add(Dense(hidden_units_1, activation=activation, kernel_initializer='normal', name='middle_1'))
    model.add(Dropout(droupout1))
    model.add(Dense(hidden_units_2, activation=activation, kernel_initializer='normal', name='middle_2'))
    model.add(Dropout(droupout2))
    model.add(Dense(output_size, activation='softmax', kernel_initializer='normal', name='Output'))
    
    Adam = optimizers.Adam(learning_rate=lr, decay=decay)
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])
    return model

model_three_layers = three_layer_model()

def classification_model(   hidden_layers = 0, 
                            hidden_units = [0],
                            hidden_activation = 'sigmoid',
                            kernel_initializer= 'random_normal',
                            dropout_rate = 0.0,
                            regularizer = None,
                            regularizer_rate = 0.001,
                            bias_initializer = 'zeros',
                            use_batch_normalization=False
                            ):
    """ Defines a neural network model for classification.      
        @param hidden_layers             Number of hidden layers
        @param hidden_units              Array of neurons per layers. The position in the array corresponds to the number of the hiddel layer.
        @param hidden_activation         Activation function used in hidden layers
        @param kernel_initializer        Initializer of synaptic weights
        @param dropout_rate              Rate of the dropout layer added after each dense layer
        @param regularizer               Type of regularizer to use, values supported are None, 'L1' or 'L2'
        @param regularizer_rate          Regularizer coefficient
        @param bias_initializer          Initializer for bias weights
        @param use_batch_normalization   Determines whether to use Batch Normalization between hidden layers or not
        @return Keras neural network or model instance
    """

    if regularizer == 'L1':
        kernel_regularizer = keras.regularizers.l1(regularizer_lambda)
    elif regularizer == 'L2':
        kernel_regularizer = keras.regularizers.l2(regularizer_lambda)
    else:
        kernel_regularizer = None

    # Create two input layers for the categorical and the numerical variables
    x1 = keras.layers.Input(shape=(5, ))
    x2 = keras.layers.Input(shape=(1, ))

    # Create an embedding layer with the categorical variable and create
    # a new input layer for the neural network, combining both the embedding and the 
    # numerical variable input layer
    embedding = keras.layers.Embedding(4, 2, input_length=1, embeddings_initializer='normal')(x2)
    flatten = keras.layers.Flatten()(embedding)
    x = keras.layers.Concatenate()([x1, flatten])
    current_layer = x

    # Add the hidden layers to the neural network
    layers = []
    for layer_index in range(hidden_layers):
        # Create a dense layer and add it to the neural network
        layers_per_layer = 1 + (1 if dropout_rate else 0) + (1 if use_batch_normalization else 0)
        previous_layer = layers[layers_per_layer * layer_index - 1] if layer_index else x
        current_layer = keras.layers.Dense(units=units_per_layer, 
                                           activation=hidden_layer_activation, 
                                           kernel_initializer=kernel_initializer,
                                           bias_initializer=bias_initializer,
                                           kernel_regularizer=kernel_regularizer
                                          )(previous_layer)
        layers.append(current_layer)
        
        # Create a dropout layer aftear each dense layer
        if dropout_rate:
            previous_layer = current_layer
            current_layer = keras.layers.Dropout(dropout_rate)(previous_layer)
            layers.append(current_layer)
        
        # Create a batch normalization layer after each dense layer
        if use_batch_normalization:
            previous_layer = current_layer
            current_layer = keras.layers.BatchNormalization()(previous_layer)
            layers.append(current_layer)

    # Add the output layer
    y = keras.layers.Dense(units=1,
                           activation='linear',
                           kernel_initializer='random_normal',
                           kernel_regularizer=kernel_regularizer
                          )(current_layer)

    # Create the neural network model
    return keras.Model(inputs=[x1, x2], outputs=y)
