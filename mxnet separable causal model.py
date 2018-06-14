
"""
Jon Stewart
Code is not particularly optimized. Basic experiment.
Does not work in all versions of mxnet. 
Model combines separable convolutions and causal convolutions (asymmetric padding, so that model cannot rely on future data.)
 Performs similarly to dilated convolutions.
"""

def Separable_asymm_layer(data, num_in_channel, num_out_channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None,  output_channel = None, suffix='', depth_mult=1, withBn=True, bn_mom=0.9, workspace=256):
    assert name is not None
    assert dilate is not None
    assert output_channel is not None
    assert isinstance(data, mx.symbol.Symbol)
    zero = mx.symbol.Variable(name=name+"-zero")
    concat = mx.symbol.Concat(*[data, zero], dim=3, name=name+"-concat")   #asymmetric 0-padding
    
    channels = mx.symbol.SliceChannel(data=concat, axis=1, num_outputs=num_in_channel) 
    channel_convs = [mx.symbol.Convolution(data=channels[i], num_filter=depth_mult, kernel=kernel, 
                           stride=stride, pad=pad, name=name+'_depthwise_kernel_'+str(i), workspace=workspace)
                           for i in range(num_in_channel)]
    depth_convs = mx.symbol.Concat(*channel_convs)
    
    pointwise_out = Conv(data=depth_convs, num_filter=num_out_channel, name=name+'point', withBn=False, bn_mom=0.9, workspace=256)
    if withBn:
        pointwise_out = mx.symbol.BatchNorm(data=pointwise_out, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    return pointwise_out

def Separable_1d_model():
    
    data = mx.sym.Variable('data')

    x1 = Separable_asymm_layer(data, num_in_channel = 3, num_out_channel = 3, kernel = (1,2),  name = 'layer1')
    
    x1a = mx.symbol.Convolution(data=x1, kernel=kernel, stride=stride,  num_filter=6, name="gatingx1")
    x1b = mx.symbol.Convolution(data=x1, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid1a")
    x1b = mx.symbol.Activation(data=x1b, act_type="sigmoid", name="gating_sigmoid_1b")
    x1g = x1a * x1b
    
    x2 = Separable_asymm_layer(x1g, num_in_channel = 4, num_out_channel = 4, kernel = (1,4),  name = 'layer2')
    
    x2a = mx.symbol.Convolution(data=x2, kernel=kernel, stride=stride, num_filter=6, name="gatingx2")
    x2b = mx.symbol.Convolution(data=x2, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid2a")
    x2b = mx.symbol.Activation(data=x2b, act_type="sigmoid", name="gating_sigmoid_2b")
    x2g = x2a * x2b
    
    x3 = Separable_asymm_layer(x2g, num_in_channel = 3, num_out_channel = 4, kernel = (1,8),  name = 'layer3')
    
    x3a = mx.symbol.Convolution(data=x3, kernel=kernel, stride=stride, num_filter=12, name="gatingx3")
    x3b = mx.symbol.Convolution(data=x3, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid3a")
    x3b = mx.symbol.Activation(data=x3b, act_type="sigmoid", name="gating_sigmoid_3b")
    x3g = x3a * x3b
    
    x4 = Separable_asymm_layer(x3g, num_in_channel = 3, num_out_channel = 4, kernel = (1,16),  name = 'layer4')
    
    x4a = mx.symbol.Convolution(data=x4, kernel=kernel, stride=stride, num_filter=6, name="gatingx4")
    x4b = mx.symbol.Convolution(data=x4, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid4a")
    x4b = mx.symbol.Activation(data=x4b, act_type="sigmoid", name="gating_sigmoid_4b")
    x4g = x4a * x4b
    
    x5 = Separable_asymm_layer(x4g, num_in_channel = 3, num_out_channel = 4, kernel = (1,8),  name = 'layer5')
    
    x5a = mx.symbol.Convolution(data=x5, kernel=kernel, stride=stride, num_filter=6, name="gatingx5)
    x5b = mx.symbol.Convolution(data=x5, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid5a")
    x5b = mx.symbol.Activation(data=x5b, act_type="sigmoid", name="gating_sigmoid_5b")
    x5g = x5a * x5b
    
    x6 = Separable_asymm_layer(x4g, num_in_channel = 3, num_out_channel = 4, kernel = (1,8),  name = 'layer7')
    
    x6a = mx.symbol.Convolution(data=x6, kernel=kernel, stride=stride, num_filter=6, name="gatingx6)
    x6b = mx.symbol.Convolution(data=x6, kernel=kernel, stride=stride, dilate=dilate, num_filter=6, name="gating_sigmoid6a")
    x6b = mx.symbol.Activation(data=x6b, act_type="sigmoid", name="gating_sigmoid_6b")
    
    x7 = mx.symbol.Convolution(data=x6b, kernel = (1,1), num_filter = 6, name = "layer7')
    out = mx.symbol.SoftmaxOutput(data=x7, name="softmax", multi_output=True)
    
    