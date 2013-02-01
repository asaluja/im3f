function [jCells] = jaggedCell(ids, numEntries)
% JAGGEDCELL Jagged cell indexing of an array.
%
% Usage:
%    [jCells] = jaggedCell(ids, numEntries)
%
% Inputs:
%    ids - array of integer ids
%    numEntries - Maximum id accounted for in ids
%
% Outputs:
%    jCells - Jagged cell array where the ith entry is an array of indices
%             corresponding to each appearance of i in ids

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

fprintf('Running jaggedCell\n');

numExamples = length(ids);
% Count the number of occurrences of each id in ids
counts = accumarrayMex(ids, 1, [numEntries,1]);
% Initialize jagged cell array
jCells = cell(size(counts));
for r = 1:numEntries
    jCells{r} = zeros(1,counts(r), 'uint32');
    counts(r) = 1;
end
% Populate jagged cell array with indices of entries
% referencing each id in ids
for e = 1:numExamples
    r = ids(e);
    jCells{r}(counts(r)) = e;
    counts(r) = counts(r) + 1; 
end

% -----------------------------END OF CODE-------------------------------
