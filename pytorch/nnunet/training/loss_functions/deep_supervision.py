#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            # Modif by PL
            # weights = [1] * len(x)
            # We want the sum of the weights to be equals to 1
            # Warning : This implementation is only relevent if we do ds. If not it will only divide the loss
            # by the number of localisers you have in your architecture (deepness of the network in this case)
            weights = [1 / len(x)] * len(x)

        else:
            weights = self.weight_factors

        # Modif by PL
        # l = weights[0] * self.loss(x[0], y[0])
        #Deep supervision with the localisers output
        l = 0
        for i in range(0, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y[0])
        return l
