# -*- coding: utf-8 -*-
"""


@author: jws0258
"""

def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5

def wavenet1(inputshape):
    
    input = Input(shape = inputshape, name = 'input')
    
    input1 = GaussianDropout(.03)(input)
    
    #res block
    
    tanh_out = Conv1D(3, atrous_rate=2, padding = 'causal', name = 'a1')(input1)#use gated convolution here
    tanh_out1 = BatchNormalization()(tanh_out)
    tanh_out2 = Activation('tanh')(tanh_out) #apply gated convolution
    tanh_out3 = Multiply()([tanh_out1, tanh_out2])
    
    sig_out = Conv1D(3, atrous_rate=2, padding = 'causal', name = 'a2')(input1)
    sig_out1 = BatchNormalization()(sig_out)
    sig_out2 = Activation('sigmoid')(sig_out) 
    sig_out3 = Multiply()([sig_out1, sig_out2])
    
    Merge1 = Multiply()([sigout3, tanh_out3])
    Res1 = Conv1D(3, atrous_rate=2, padding = 'causal', name = 'a3')(Merge1)
    Res1 = BatchNormalization()(Res1)
    Merge2 = Activation('tanh')(Res1)
    Merge2 = Multiply()([Res1, Merge2])
    Merge2a = Add()([Merge2, input1])
    
    tanh_out = Conv1D(3, atrous_rate=4, padding = 'causal', name = 'b1')(Merge2a)
    tanh_out1 = BatchNormalization()(tanh_out)
    tanh_out2 = Activation('tanh')(tanh_out) 
    tanh_out3 = Multiply()([tanh_out1, tanh_out2])
    
    sig_out = Conv1D(3, atrous_rate=4, padding = 'causal', name = 'b2')(Merge2a)
    sig_out1 = BatchNormalization()(sig_out)
    sig_out2 = Activation('sigmoid')(sig_out) 
    sig_out3 = Multiply()([sig_out1, sig_out2])
    
    Merge1 = Multiply()([sigout3, tanh_out3])
    Res1 = Conv1D(3, atrous_rate=4, padding = 'causal', name = 'b3')(Merge1)
    Res1 = BatchNormalization()(Res1)
    Merge2 = Activation('tanh')(Res1)
    Merge2 = Multiply()([Res1, Merge2])
    Merge2b = Add()([Merge2a, Merge2])
    
    tanh_out = Conv1D(3, atrous_rate=8, padding = 'causal', name = 'c1')(Merge2b)
    tanh_out1 = BatchNormalization()(tanh_out)
    tanh_out2 = Activation('tanh')(tanh_out)
    tanh_out3 = Multiply()([tanh_out1, tanh_out2])
    
    sig_out = Conv1D(3, atrous_rate=8, padding = 'causal', name = 'c2')(Merge2b)
    sig_out1 = BatchNormalization()(sig_out)
    sig_out2 = Activation('sigmoid')(sig_out) 
    sig_out3 = Multiply()([sig_out1, sig_out2])
    
    Merge1 = Multiply()([sigout3, tanh_out3])
    Res1 = Conv1D(3, atrous_rate=8, padding = 'causal', name = 'c3')(Merge1)
    Res1 = BatchNormalization()(Res1)
    Merge2 = Activation('tanh')(Res1)
    Merge2 = Multiply()([Res1, Merge2])
    Merge2c = Add()([Merge2b, Merge2])
    
    tanh_out = Conv1D(3, atrous_rate=16, padding = 'causal', name = 'd1')(Merge2c)
    tanh_out1 = BatchNormalization()(tanh_out)
    tanh_out2 = Activation('tanh')(tanh_out) #apply gated convolution
    tanh_out3 = Multiply()([tanh_out1, tanh_out2])
    
    sig_out = Conv1D(3, atrous_rate=16, padding = 'causal', name = 'd2')(Merge2c)
    sig_out1 = BatchNormalization()(sig_out)
    sig_out2 = Activation('sigmoid')(sig_out) #apply gated convolution
    sig_out3 = Multiply()([sig_out1, sig_out2])
    
    Merge1 = Multiply()([sigout3, tanh_out3])
    Res1 = Conv1D(3, atrous_rate=16, padding = 'causal', name = 'd3')(Merge1)
    Res1 = BatchNormalization()(Res1)
    Merge2 = Activation('tanh')(Res1)
    Merge2 = Multiply()([Res1, Merge2])
    Merge2d = Add()([Merge2c, Merge2])
    
    tanh_out = Conv1D(3, atrous_rate=32, padding = 'causal', name = 'e1')(Merge2d)
    tanh_out1 = BatchNormalization()(tanh_out)
    tanh_out2 = Activation('tanh')(tanh_out) #apply gated convolution
    tanh_out3 = Multiply()([tanh_out1, tanh_out2])
    
    sig_out = Conv1D(3, atrous_rate=32, padding = 'causal', name = 'e2')(Merge2d)
    sig_out1 = BatchNormalization()(sig_out)
    sig_out2 = Activation('sigmoid')(sig_out) 
    sig_out3 = Multiply()([sig_out1, sig_out2])
    
    Merge1 = Multiply()([sigout3, tanh_out3])
    Res1 = Conv1D(3, atrous_rate=32, padding = 'causal', name = 'e3')(Merge1)
    Res1 = BatchNormalization()(Res1)
    Merge2 = Activation('tanh')(Res1)
    Merge2 = Multiply()([Res1, Merge2])
    Merge2b = Add()([Merge2d, Merge2])
    
    xout = Flatten()(Merge2b)
    xout = Dense(1, activation = 'linear', name = 'out1')(Merge2b)
    model = Model(input, xout)
    ADM = Adam(lr==0.0015, beta_1=0.9, beta_2=0.999, decay=0.00001, clipnorm=1.)
    model.compile(loss=RMSLE, optimizer=ADM) #custom loss function above
    return model
