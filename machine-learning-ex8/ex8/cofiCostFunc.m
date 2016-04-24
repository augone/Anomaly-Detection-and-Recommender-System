function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
prediction = X * Theta';
diff = prediction - Y;
J = 0.5*sum(sum(diff(R == 1).^2));
costRegularizationTerm = lambda/2*(sum(sum(X.^2))+sum(sum(Theta.^2)));
J = J + costRegularizationTerm;


% for iterator1 = 1 : num_movies
%     for iterator2 = 1 : num_features
%         for iterator3 = 1 : num_users
%             if not (R(iterator1,iterator3) == 0)
%                 X_grad(iterator1,iterator2) = X_grad(iterator1,iterator2)+...
%                     (Theta(iterator3,:) * X(iterator1,:)' - Y(iterator1,iterator3))...
%                         *Theta(iterator3,iterator2);
%             end
%         end
%     end
% end
for iterator1 = 1 : num_movies
    idx = find(R(iterator1,:) == 1);
    Theta_temp = Theta(idx,:);
    Y_temp = Y(iterator1,idx);
    X_grad(iterator1,:)  = (X(iterator1,:)*Theta_temp' - Y_temp)* Theta_temp;
    XregularizaitonTerm = lambda*X(iterator1,:);
    X_grad(iterator1,:) = X_grad(iterator1,:) + XregularizaitonTerm;
end


for iterator1 = 1: num_users
    idx = find(R(:,iterator1) == 1);
    Xtemp = X(idx,:);
    Ytemp = Y(idx,iterator1);
    Theta_grad(iterator1,:) = (Theta(iterator1,:)*Xtemp' - Ytemp')*Xtemp;
    ThetaRegularizaitonTerm = lambda*Theta(iterator1,:);
    Theta_grad(iterator1,:) = Theta_grad(iterator1,:) + ThetaRegularizaitonTerm;
end

    
    
% for iterator1 = 1 : num_users
%     for iterator2 = 1 : num_features
%         for iterator3 = 1 : num_movies
%             if not (R(iterator3,iterator1) == 0)
%                 Theta_grad(iterator1,iterator2) =  Theta_grad(iterator1,iterator2)...
%                     +( sum(Theta(iterator1,:).* X(iterator3,:)) - Y(iterator3,iterator1))*X(iterator3,iterator2);
%             end
%         end
%     end
% end
% 

















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
