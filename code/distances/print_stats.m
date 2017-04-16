function [res, threshold] = print_stats(D, k)
% Returns some stats for the distance matrix.
% This is to help find the corresponding distance threshold for a given
% mean number of neighbours.
%
% The first return value is a table with thresholds and average number of
% neighbours.
% The second return value is the distance threshold to use if we want on
% average 'k' number of neighbours (second argument).
%
% G.Sfikas June 2016
N = size(D, 1);
Dmean = mean(D(D(:)~=0));
Dstd = std(D(D(:)~=0));
j = 1;
%fprintf('distance threshold\taverage neighbours\n');
for i = .01*Dstd:.01*Dstd:Dstd
    thres = i*Dmean;
    Dnew = zeros(size(D));
    Dnew(D < thres) = 1;
    Dnew_sum = sum(Dnew, 2);
    neighbours = mean(Dnew_sum(:));
    res(j, 1) = thres;
    res(j, 2) = neighbours-1;
    if(exist('k', 'var') && j > 1 && res(j-1, 2) < k)
        threshold = thres;
    end
    %fprintf('%d\t\t\t%2.2f\n', res(j, 1), res(j, 2));    
    j = j + 1;
    if(neighbours == N)
        break;
    end
end

return;