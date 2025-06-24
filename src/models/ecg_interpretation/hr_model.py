import tensorflow as tf


class RhythmModels:
    @staticmethod
    def conv1d_net(x,
                   num_filters,
                   kernel_size,
                   strides=1,
                   pad='SAME',
                   act=True,
                   bn=True,
                   rate=0.5,
                   name=""):
        """

        """
        if bn:
            x = tf.keras.layers.BatchNormalization(axis=-1, name=name + '_bn')(x)

        if act:
            x = tf.keras.layers.ReLU(name=name + '_act')(x)

        if rate < 1.0:
            x = tf.keras.layers.Dropout(rate=rate, name=name + '_drop')(x)

        x = tf.keras.layers.Conv1D(filters=num_filters,
                                   kernel_size=kernel_size,
                                   strides=strides,
                                   padding=pad,
                                   name=name + '_conv1d')(x)

        return x

    def block1d_loop(self, xx, ff, stage, step):
        """

            :param xx:
            :param ff:
            :param stage:
            :param step:
            :return:
            """
        xx_skip = xx
        f1, f2 = ff
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        xx = self.conv1d_net(x=xx,
                             num_filters=f1,
                             kernel_size=3,
                             strides=1,
                             pad='SAME',
                             act=True,
                             bn=True,
                             rate=0.5,
                             name="resnet11a_{}_{}".format(step, stage))
        # Batch norm, Activation, Dropout, Convolution (stride=1)
        xx = self.conv1d_net(x=xx,
                             num_filters=f2,
                             kernel_size=3,
                             strides=1,
                             pad='SAME',
                             act=True,
                             bn=True,
                             rate=0.5,
                             name="resnet11b_{}_{}".format(step, stage))

        xx = tf.keras.layers.Add(name="skip11_{}_{}".format(step, stage))([xx, xx_skip])
        return xx

    def regression_conv_net(self,
                            feature_len,
                            filters_rhythm_net=None,
                            num_loop=7,
                            rate=0.5,
                            name='regression_conv_net'):
        """

        """
        if filters_rhythm_net is None:
            filters_rhythm_net = [(16, 16),
                                  (16, 32),
                                  (32, 48),
                                  (48, 64)]
        else:
            tmp = []
            for i, f in enumerate(filters_rhythm_net):
                tmp.append((f - (i * filters_rhythm_net[0]), f))

            filters_rhythm_net = tmp.copy()

        input_layer = tf.keras.layers.Input(shape=(feature_len,))
        resnet_input_layer = tf.keras.layers.Reshape((feature_len, 1))(input_layer)
        # Convolution(stride=2)
        x = self.conv1d_net(x=resnet_input_layer,
                            num_filters=filters_rhythm_net[0][0],
                            kernel_size=3,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="input_stage")

        for st, ff in enumerate(filters_rhythm_net):
            st += 1
            f1, f2 = ff
            name = 'stage_{}'.format(st)
            # 1x1 Convolution (stride=2)
            x_skip = self.conv1d_net(x=x,
                                     num_filters=f2,
                                     kernel_size=1,
                                     strides=2,
                                     pad='SAME',
                                     act=False,
                                     bn=False,
                                     rate=1.0,
                                     name="skip12_" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=2)
            x = self.conv1d_net(x=x,
                                num_filters=f1,
                                kernel_size=3,
                                strides=2,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet12" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=1)
            x = self.conv1d_net(x=x,
                                num_filters=f2,
                                kernel_size=3,
                                strides=1,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet11" + name)

            x = tf.keras.layers.Add(name="add_" + name)([x, x_skip])
            ffs = [(f2, f2) for _ in range(num_loop)]
            for sl, ffl in enumerate(ffs):
                x = self.block1d_loop(x, ffl, name, sl)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(input_layer, x, name=name)

    def regression_conv_seq(self,
                            feature_len,
                            filters_rhythm_net=None,
                            num_loop=7,
                            rate=0.5,
                            name='regression_conv_seq'):
        """

        """
        if filters_rhythm_net is None:
            filters_rhythm_net = [(16, 16),
                                  (16, 32),
                                  (32, 48),
                                  (48, 64)]
        else:
            tmp = []
            for i, f in enumerate(filters_rhythm_net):
                tmp.append((f - (i * filters_rhythm_net[0]), f))

            filters_rhythm_net = tmp.copy()

        input_layer = tf.keras.layers.Input(shape=(feature_len,))
        resnet_input_layer = tf.keras.layers.Reshape((feature_len, 1))(input_layer)
        # Convolution(stride=2)
        x = self.conv1d_net(x=resnet_input_layer,
                            num_filters=filters_rhythm_net[0][0],
                            kernel_size=3,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="input_stage")

        for st, ff in enumerate(filters_rhythm_net):
            st += 1
            f1, f2 = ff
            name = 'stage_{}'.format(st)
            # 1x1 Convolution (stride=2)
            x_skip = self.conv1d_net(x=x,
                                     num_filters=f2,
                                     kernel_size=1,
                                     strides=2,
                                     pad='SAME',
                                     act=False,
                                     bn=False,
                                     rate=1.0,
                                     name="skip12_" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=2)
            x = self.conv1d_net(x=x,
                                num_filters=f1,
                                kernel_size=3,
                                strides=2,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet12" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=1)
            x = self.conv1d_net(x=x,
                                num_filters=f2,
                                kernel_size=3,
                                strides=1,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet11" + name)

            x = tf.keras.layers.Add(name="add_" + name)([x, x_skip])
            ffs = [(f2, f2) for _ in range(num_loop)]
            for sl, ffl in enumerate(ffs):
                x = self.block1d_loop(x, ffl, name, sl)

        lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(x.shape[-1], return_sequences=False, dropout=rate))(x)

        x = tf.keras.layers.Dense(1, activation='linear')(lstm_layer)
        return tf.keras.Model(input_layer, x, name=name)

    def classification_conv_net(self,
                                feature_len,
                                num_output=1,
                                filters_rhythm_net=None,
                                num_loop=7,
                                rate=0.5,
                                name='classification_conv_net'):
        """

        """
        if filters_rhythm_net is None:
            filters_rhythm_net = [(16, 16),
                                  (16, 32),
                                  (32, 48),
                                  (48, 64)]
        else:
            tmp = []
            for i, f in enumerate(filters_rhythm_net):
                tmp.append((f - (i * filters_rhythm_net[0]), f))

            filters_rhythm_net = tmp.copy()

        input_layer = tf.keras.layers.Input(shape=(feature_len,))
        resnet_input_layer = tf.keras.layers.Reshape((feature_len, 1))(input_layer)
        # Convolution(stride=2)
        x = self.conv1d_net(x=resnet_input_layer,
                            num_filters=filters_rhythm_net[0][0],
                            kernel_size=3,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="input_stage")

        for st, ff in enumerate(filters_rhythm_net):
            st += 1
            f1, f2 = ff
            name = 'stage_{}'.format(st)
            # 1x1 Convolution (stride=2)
            x_skip = self.conv1d_net(x=x,
                                     num_filters=f2,
                                     kernel_size=1,
                                     strides=2,
                                     pad='SAME',
                                     act=False,
                                     bn=False,
                                     rate=1.0,
                                     name="skip12_" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=2)
            x = self.conv1d_net(x=x,
                                num_filters=f1,
                                kernel_size=3,
                                strides=2,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet12" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=1)
            x = self.conv1d_net(x=x,
                                num_filters=f2,
                                kernel_size=3,
                                strides=1,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet11" + name)

            x = tf.keras.layers.Add(name="add_" + name)([x, x_skip])
            ffs = [(f2, f2) for _ in range(num_loop)]
            for sl, ffl in enumerate(ffs):
                x = self.block1d_loop(x, ffl, name, sl)

        logits_layer = tf.keras.layers.Dense(num_output)(x)
        softmax_layer = tf.keras.layers.Softmax(axis=-1)(logits_layer)
        return tf.keras.Model(input_layer, softmax_layer, name=name)

    def classification_conv_seq(self,
                                feature_len,
                                num_output=1,
                                filters_rhythm_net=None,
                                num_loop=7,
                                rate=0.5,
                                name='classification_conv_seq'):
        """

        """
        if filters_rhythm_net is None:
            filters_rhythm_net = [(16, 16),
                                  (16, 32),
                                  (32, 48),
                                  (48, 64)]
        else:
            tmp = []
            for i, f in enumerate(filters_rhythm_net):
                tmp.append((f - (i * filters_rhythm_net[0]), f))

            filters_rhythm_net = tmp.copy()

        input_layer = tf.keras.layers.Input(shape=(feature_len,))
        resnet_input_layer = tf.keras.layers.Reshape((feature_len, 1))(input_layer)
        # Convolution(stride=2)
        x = self.conv1d_net(x=resnet_input_layer,
                            num_filters=filters_rhythm_net[0][0],
                            kernel_size=3,
                            strides=2,
                            pad='SAME',
                            act=False,
                            bn=False,
                            rate=1.0,
                            name="input_stage")

        for st, ff in enumerate(filters_rhythm_net):
            st += 1
            f1, f2 = ff
            name = 'stage_{}'.format(st)
            # 1x1 Convolution (stride=2)
            x_skip = self.conv1d_net(x=x,
                                     num_filters=f2,
                                     kernel_size=1,
                                     strides=2,
                                     pad='SAME',
                                     act=False,
                                     bn=False,
                                     rate=1.0,
                                     name="skip12_" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=2)
            x = self.conv1d_net(x=x,
                                num_filters=f1,
                                kernel_size=3,
                                strides=2,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet12" + name)
            # Batch norm, Activation, Dropout, Convolution (stride=1)
            x = self.conv1d_net(x=x,
                                num_filters=f2,
                                kernel_size=3,
                                strides=1,
                                pad='SAME',
                                act=True,
                                bn=True,
                                rate=rate,
                                name="resnet11" + name)

            x = tf.keras.layers.Add(name="add_" + name)([x, x_skip])
            ffs = [(f2, f2) for _ in range(num_loop)]
            for sl, ffl in enumerate(ffs):
                x = self.block1d_loop(x, ffl, name, sl)

        lstm_layer = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(x.shape[-1], return_sequences=False, dropout=rate))(x)

        logits_layer = tf.keras.layers.Dense(num_output)(lstm_layer)
        softmax_layer = tf.keras.layers.Softmax(axis=-1)(logits_layer)
        return tf.keras.Model(input_layer, softmax_layer, name=name)


def load_hr_model(checkpoint_dir):
    rhythm_model = RhythmModels()
    my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    tf.config.experimental.set_visible_devices(devices=my_devices, device_type='CPU')

    feature_len = 2500
    model_name = "regression_conv_net"
    test_model = getattr(rhythm_model, model_name)(feature_len)
    test_model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
    # print(test_model.summary())
    print('-----Model_loaded----')

    return test_model


if __name__ == "__main__":
    import numpy as np

    model = load_hr_model(checkpoint_dir='./checkpoints/HR_model_best_MAE')
    x = np.random.random((1, 2500))
    with tf.device("/cpu:0"):
        hr = model.predict(x)
    print(hr)


