function [ G, Dmat ] = makeKnnG(Y, k)
%MAKEKNNG Construct a symmetric k-nearest neighbor adjacency matrix using
%Euclidean distance.
%
%Input:
%   - Y: DxN matrix where D is the dimensions and N is the data size.
%   - k: k in kNN
%Output: 
%   - G: NxN matrix G of {0,1}
%   - Dmat: NxN pairwise distance matrix.
%
% By default, the returned G has 0 diagonal i.e., no self neighbouring.
%
% @author Wittawat. Created on 19 May 2015.
%

[d, n] = size(Y);
assert(isscalar(k));
if k >= n
    error('There are at most n-1 neighbours.');
end

y2sum = sum(Y.^2, 1);
% NxN. 0 diagonal.
Dmat = sqrt( bsxfun(@plus, y2sum',  y2sum) - 2*(Y'*Y) );
G = convertD2G(Dmat, k);

end
