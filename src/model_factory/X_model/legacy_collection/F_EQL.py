# class f_eql(nn.Module):
#     def __init__(self, input_channel=1, bias=False, symbolic_bases=None, scale=...,
#      skip_connect=True, down_sampling_kernel=None, down_sampling_stride=2,
#       num_class=4, device='cuda', amount=0.5) -> None:
#         super(f_eql,self).__init__()
        
#         self.input_channel = input_channel
#         self.bias = bias
#         self.skip_connect = skip_connect
#         self.amount = amount
        
#         # assert len(symbolic_bases) == len(scale)
#         self.scale = scale
#         self.symbolic_bases = feature_base()
#         self.symbolic_bases_4reg = symbolic_base(['mul','sin','exp','idt','sig','tanh','pi','e'])
#         self.down_sampling_kernel = down_sampling_kernel
#         if self.down_sampling_kernel is not None :
#             assert len(scale) == len(self.down_sampling_kernel), 'dimention mismatch'
#         self.down_sampling_stride = down_sampling_stride
#         self.device = device
#         # 符号变换层
        
#         self.symbolic_transform_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases],
#                        input_channel= self.input_channel,
#                        layer_type = 'transform')
        
#         # 如果有降采样 则倒数第二层
#         final_dim = self.symbolic_transform_layer[-1].output_channel
        
#         # 符号回归层
        
#         self.symbolic_regression_layer = self.__make_layer__(symbolic_bases = [self.symbolic_bases_4reg], # 1层符号回归层
#                        input_channel= final_dim,
#                        layer_type = 'regression')
        
#         final_dim = self.symbolic_regression_layer[-1].output_channel
        
#         # 线性组合
        
#         self.regression_layer = nn.Linear(final_dim,num_class,bias = bias)
        
#         self.to(self.device)
#     def __make_layer__(self,
#                        symbolic_bases,
#                        input_channel = 1,
#                        layer_type = 'transform' # 'regression'
#                        ):            
#         layers = []
#         layer_selection = neural_symbolc_base if layer_type == 'transform' else neural_symbolic_regression
        
#         for i,symbolic_base in enumerate(symbolic_bases):
                
                            
#             next_channel = layer.output_channel if i else input_channel
            
#             layer = layer_selection( symbolic_base = symbolic_base,
#                 #  initialization = None,
#                  input_channel = next_channel, 
#                 #  output_channel = None,
#                  bias = self.bias,
#                  device = self.device,
#                  scale = 1,
#                 #  kernel_size = 1,
#                 #  stride = 1,
#                  skip_connect = self.skip_connect,
#                  amount = self.amount)            
#             layers.append(layer)

                
#         return nn.ModuleList(layers)
    
#     def norm(self,x):
#         mean = x.mean(dim = 1,keepdim = True)
#         std = x.std(dim = 1,keepdim = True)
#         out = (x-mean)/(std + 1e-10)
#         return out        
    
#     def forward(self,x):
#         for layer in self.symbolic_transform_layer:
#             x = layer(x)  # 直接nan
#         self.feature = x.mean(dim=-1, keepdim = True)
#         x = self.feature
#         for layer in self.symbolic_regression_layer:
#             x = self.norm(x)
#             x = layer(x)
            
#         x = x.squeeze()
#         x = self.regression_layer(x)
#         # x = nn.Softmax(dim=1)(x)
#         return x