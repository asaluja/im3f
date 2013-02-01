
% -----------------------------------------------------------------------     
%
% Last revision: 9-July-2010
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
% setenv OMP_NUM_THREADS 1;

%if matlabpool('size') == 0
%    matlabpool;
%end
experDir = '~/im3f/data/experiments';
% m3f_tib_exper(experDir, experName, dataName, splitNames, topicModel, initMode, seed, numFacs, KU, KM, gamma, beta, T)
% topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}

%for gamma = [0.01, 0.1, 1]
%for gamma = 0.01
%    for beta = [0.01, 0.1, 1]
%        m3f_tib_exper(experDir, 'crf', 'ml100k', {'a1'}, 'crf', 'collapsed', 3, 12345, 40, 1, 1, gamma, beta, 100);
%    end
%end
matlabpool('open',2);
GS={'collapsed' 'noncollapsed'};
KUKM=[2 1];
beta=[1 0.1];
spmd
    idx = labindex; 
    m3f_tib_exper(experDir, 'crf', 'ml100k', {'a1'}, 'crf', char(GS(idx)), 3, 12345, 40, KUKM(idx), KUKM(idx), 0.1, beta(idx), 100);
    %m3f_tib_exper(experDir, 'crf', 'ml100k', {'a1'}, 'crf','noncollapsed', 3, 12345, 40, 1, 1, 0.1, 0.1, 100);
end
