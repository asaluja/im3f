function [res] = evalPreds(groundTruth, preds, metric) 
% EVALPREDS Evaluate predictions under a given error metric.
%
% Usage:
%    [res] = evalPreds(groundTruth, preds, metric) 
%
% Inputs:
%    groundTruth - True values
%    preds - Predicted values
%    metric - Name of error metric to apply
%             'rmse' => Root mean squared error
%             'mae' => Mean absolute error
%
% Outputs:
%    res - Error of predictions under given metric.

% -----------------------------------------------------------------------     
%
% Last revision: 2-July-2010
%
% Authors: Lester Mackey and David Weiss
% License: MIT License
%
% Copyright (c) 2010 Lester Mackey & David Weiss
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the
% "Software"), to deal in the Software without restriction, including
% without limitation the rights to use, copy, modify, merge, publish,
% distribute, sublicense, and/or sell copies of the Software, and to
% permit persons to whom the Software is furnished to do so, subject to
% the following conditions:
% 
% The above copyright notice and this permission notice shall be
% included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
% EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
% MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
% NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
% LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
% OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
% WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
%
% -----------------------------------------------------------------------

% -----------------------------BEGIN CODE--------------------------------

res = -1;
if(strcmp(metric, 'rmse'))
    res = sqrt(mean((groundTruth - preds).^2));
elseif(strcmp(metric, 'mae'))
    res = mean(abs(groundTruth - preds));
end

% -----------------------------END OF CODE-------------------------------
