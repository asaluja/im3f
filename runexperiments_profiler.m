function runexperiments_profiler(sampler, init_num_topics, beta, prof_results)
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

experDir = '~/im3f/data/experiments';
% m3f_tib_exper(experDir, experName, dataName, splitNames, topicModel, initMode, seed, numFacs, KU, KM, gamma, beta, T)
% topicModel - {shcrp="Shared CRP", secrp="Separate CRP", crf="CRF"}
profile on;
m3f_tib_exper(experDir, 'crf', 'ml100k', {'a1'}, 'crf', char(sampler), 3, 12345, ...
              40, str2num(init_num_topics), str2num(init_num_topics), 0.1, str2num(beta), 100);
profsave(profile('info'), char(prof_results)); 
exit
