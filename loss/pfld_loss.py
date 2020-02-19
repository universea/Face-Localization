import paddle.fluid as fluid
import math

class Loss():
    def __init__(self, scale=1.0):
        self.scale = scale

    def PFLDLoss(self,attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize=1):
        # landmark_gt[196] 
        # attribute_gt[6] 
        # euler_angle_gt[3]
 
        print("attribute_gt.shape",attribute_gt.shape)
        print("landmark_gt.shape",landmark_gt.shape)
        print("euler_angle_gt.shape",euler_angle_gt.shape)
        print("angle.shape",angle.shape)
        print("landmarks.shape",landmarks.shape)
        print("train_batchsize",train_batchsize)
       
        weight_angle = fluid.layers.reduce_sum(1-fluid.layers.cos(angle - euler_angle_gt),dim=1)
        print("weight_angle",weight_angle.shape)

        attributes_w_n = attribute_gt[:, 1:6]
        print("attributes_w_n",attributes_w_n.shape)

        
        mat_ratio = fluid.layers.reduce_mean(attributes_w_n, dim=0)
        print("mat_ratio",mat_ratio.shape)
        #fluid.layers.Print(mat_ratio,10, message="The content of input layer:")
        
        #mat_ratio = torch.Tensor([
        #    1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        #]).cuda()        
        # torch一行代码,paddle要经过十几行去实现, api有待完善，比如这些map_fn
        
        
        zero = fluid.layers.fill_constant(shape=[1], value=0.0, dtype='float32')
        xd0 = fluid.layers.greater_than(mat_ratio, zero) #  [1,0,0,1,1]/mat_ratio
        xd0 = fluid.layers.cast(x=xd0, dtype="float32")
        xx0 = fluid.layers.less_equal(mat_ratio, zero)#     [0,1,1,0,0]*train_batchsize
        xx0 = fluid.layers.cast(x=xx0, dtype="float32")
        train_batchsize = fluid.layers.fill_constant(shape=[1], value=train_batchsize, dtype='float32')
        mat_ratio_fix =  fluid.layers.clip(x=mat_ratio, min=1e-7, max=2.0)
        xd0 = xd0/mat_ratio_fix
        xx0 = xx0*train_batchsize
        mat_ratio = xd0 + xx0
        #fluid.layers.Print(mat_ratio,10, message="The content of input layer:")
        print("mat_ratio",mat_ratio.shape)

        am_mul = attributes_w_n*mat_ratio
        weight_attribute = fluid.layers.reduce_sum(am_mul, dim=1)
        print("weight_attribute",weight_attribute.shape)
        
        l2_distant = fluid.layers.reduce_sum(fluid.layers.square(landmark_gt - landmarks) , dim=1)
        print("l2_distant",l2_distant.shape)
        
        return fluid.layers.mean(weight_angle *weight_attribute* l2_distant), fluid.layers.mean(l2_distant)
        
        
    def smoothL1self(self,y_true, y_pred, beta = 1):
        """
        very similar to the smooth_l1_loss from pytorch, but with
        the extra beta parameter
        """
        mae = fluid.layers.abs(y_true - y_pred)
        loss = fluid.layers.reduce_sum(fluid.layers.where(mae>beta, mae-0.5*beta , 0.5*mae**2/beta), axis=-1)
        return fluid.layers.mean(loss)
    
    def mse_loss(self,y_true, y_pred):
        

        mse_loss = fluid.layers.huber_loss(input=y_pred, label=y_true, delta=1.0)
        
        mse_loss = fluid.layers.mean(mse_loss)
        
        return mse_loss

    
    def wing_loss(self,y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK = 98):

        y_pred = fluid.layers.reshape(y_pred, shape=(-1, N_LANDMARK, 2))
        y_true = fluid.layers.reshape(y_true, shape=(-1, N_LANDMARK, 2))
        
        x = y_true - y_pred
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = fluid.layers.abs(x)
        
        where = fluid.layers.less_than(x=absolute_x, y=w)
        where = fluid.layers.cast(x=where, dtype=np.uint32)
        
        T = w * fluid.layers.log(1.0 + absolute_x/epsilon)
        
        F = absolute_x - c
        
        

        
        losses = fluid.layers.where(w > absolute_x, w * fluid.layers.log(1.0 + absolute_x/epsilon), absolute_x - c)
        loss = fluid.layers.reduce_mean(torch.sum(losses, axis=[1, 2]), axis=0)
        return loss
