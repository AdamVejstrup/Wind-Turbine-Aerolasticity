clear all; close all; clc;

L=2; %[m]
m1=1; %[kg]
m2=0.5; %[kg/m]
sm=0.5*m2*L^2;
im=(m2*L^3)/3;
g=9.81;

% x(1,:) is the position of the cart
% x(2,:) is the angular position of the beam
% dx(1,:) is the velocity of the cart
% dx(2,:) is the angular velocity of beam

%% Initialization
x(1,1)=0;
x(2,1)=0;

dx(1,1)=0;
dx(2,1)=0;

gf(1,1)=0;
gf(2,1)=0;

% beta = 0.25;
% gamma = 0.50;
% 
beta = 0.27;
gamma = 0.51;

h = 0.001;   % time step
time = [0:h:50];

tf = 0.000001; %[1e-1;1e-1]; % regidual limit
tf_dis = 1e-4;

x_time(:,1) = x;  % saving displacement vector in time
dx_time(:,1) = dx;  % saving velocity vector in time

%% Mass matrix
massmatrix(1,1)=m2*L+m1;
massmatrix(1,2)=-sm*sin(x(2,1));
massmatrix(2,1)=massmatrix(1,2);
massmatrix(2,2)=im;

%% Damping matrix
dampmatrix = zeros(2,2);

%% Stiffness matrix
stiffmatrix = zeros(2,2);

%% Generalized force matrix
gf(1,1)=dx(2,1)^2*cos(x(2,1))*sm;
gf(2,1)=g*sm*cos(x(2,1));

%% step 1 for initial condition
ddx = inv(massmatrix)*(gf - dampmatrix*dx - stiffmatrix*x);

for n = 2:length(time)
    
    %% step 2 prediction step
    ddx_up = ddx;
    dx_up = dx + h*ddx;
    x_up = x + h*dx + 0.5*h^2*ddx;
        
    %% step 3 residual calculation
    k = 1;
    iter = 0;
    
    while k == 1
        
        iter = iter + 1;
        % mass and generalized force update
        massmatrix_up(1,1)=m2*L+m1;
        massmatrix_up(1,2)=-sm*sin(x_up(2,1));
        massmatrix_up(2,1)=massmatrix_up(1,2);
        massmatrix_up(2,2)=im;
        
        gf_up(1,1)=dx_up(2,1)^2*cos(x_up(2,1))*sm;
        gf_up(2,1)=g*sm*cos(x_up(2,1));
        
        if (iter==1) && (n == 2)
            disp(x_up)
        end
        
        residual = gf_up - massmatrix_up*ddx_up - dampmatrix*dx_up - stiffmatrix*x_up;
        r_max = max(abs(residual));
           
        K_eff = stiffmatrix + (gamma/(beta*h))*dampmatrix +(1/(beta*h^2))*massmatrix_up;
        delta_x = inv(K_eff) * residual;
        
        x_up = x_up + delta_x;
        dx_up = dx_up + (gamma/(beta*h))*delta_x;
        ddx_up = ddx_up + (1/(beta*h^2))*delta_x;

     
        if ( r_max > tf)  
            k = 1;

        else
            k = 0;
        end
        
        if iter > 600
            fprintf('iternation %f \n',iter);
            break
        end
    end
    
    x = x_up;
    dx = dx_up;
    ddx = ddx_up;    
    
    %% saving displacement and velocity vector in time
    x_time(:,n) = x;
    dx_time(:,n) = dx;
    
 
end

plot(time,x_time(1,:),'r','Linewidth',4); 
hold on
plot(time,x_time(2,:),'Linewidth',4);
grid on;
legend ('x','\theta');
hold on;

% clear all;
% L=2; %[m]
% m1=1; %[kg]
% m2=0.5; %[kg/m]
% sm=0.5*m2*L^2;
% im=(m2*L^3)/3;
% g=9.81;
% 
% % x(1,:) is the position of the cart
% % x(2,:) is the angular position of the beam
% % dx(1,:) is the velocity of the cart
% % dx(2,:) is the angular velocity of beam
% x(1,1)=0;
% x(2,1)=0;
% dx(1,1)=0;
% dx(2,1)=0;
% 
% massmatrix=zeros(2,2);
% 
% 
% dt=0.001;
% for n=1:50000
%  time(n)=n*dt;
% massmatrix(1,1)=m2*L+m1;
% massmatrix(1,2)=-sm*sin(x(2,n));
% massmatrix(2,1)=massmatrix(1,2);
% massmatrix(2,2)=im;
% 
% gf(1)=dx(2,n)^2*cos(x(2,n))*sm;
% gf(2)=g*sm*cos(x(2,n));
% 
% xdotdot=(gf/massmatrix)';
% 
% A(1)=xdotdot(1)*dt/2;
% A(2)=xdotdot(2)*dt/2;
% b(1)=(dx(1,n)+0.5*A(1))*dt/2;
% b(2)=(dx(2,n)+0.5*A(2))*dt/2;
% %update massmatrix and RHS
% 
% pos(1)=x(1,n)+b(1);
% pos(2)=x(2,n)+b(2);
% vel(1)=dx(1,n)+A(1);
% vel(2)=dx(2,n)+A(2);
% massmatrix(1,1)=m2*L+m1;
% massmatrix(1,2)=-sm*sin(pos(2));
% massmatrix(2,1)=massmatrix(1,2);
% massmatrix(2,2)=im;
% 
% gf(1)=vel(2)^2*cos(pos(2))*sm;
% gf(2)=g*sm*cos(pos(2));
% 
% xdotdot=(gf/massmatrix)';
% 
% B(1)=xdotdot(1)*dt/2;
% B(2)=xdotdot(2)*dt/2;
% 
% pos(1)=x(1,n)+b(1);
% pos(2)=x(2,n)+b(2);
% vel(1)=dx(1,n)+B(1);
% vel(2)=dx(2,n)+B(2);
% massmatrix(1,1)=m2*L+m1;
% massmatrix(1,2)=-sm*sin(pos(2));
% massmatrix(2,1)=massmatrix(1,2);
% massmatrix(2,2)=im;
% 
% gf(1)=vel(2)^2*cos(pos(2))*sm;
% gf(2)=g*sm*cos(pos(2));
% xdotdot=(gf/massmatrix)';
% C(1)=xdotdot(1)*dt/2;
% C(2)=xdotdot(2)*dt/2;
% 
% d(1)=(dx(1,n)+C(1))*dt;
% d(2)=(dx(2,n)+C(2))*dt;
% pos(1)=x(1,n)+d(1);
% pos(2)=x(2,n)+d(2);
% vel(1)=dx(1,n)+2*C(1);
% vel(2)=dx(2,n)+2*C(2);
% massmatrix(1,1)=m2*L+m1;
% massmatrix(1,2)=-sm*sin(pos(2));
% massmatrix(2,1)=massmatrix(1,2);
% massmatrix(2,2)=im;
% 
% gf(1)=vel(2)^2*cos(pos(2))*sm;
% gf(2)=g*sm*cos(pos(2));
% xdotdot=(gf/massmatrix)';
% D(1)=xdotdot(1)*dt/2;
% D(2)=xdotdot(2)*dt/2;
% 
% x(1,n+1)=x(1,n)+(dx(1,n)+(A(1)+B(1)+C(1))/3)*dt;
% x(2,n+1)=x(2,n)+(dx(2,n)+(A(2)+B(2)+C(2))/3)*dt;
% dx(1,n+1)=dx(1,n)+(A(1)+2*B(1)+2*C(1)+D(1))/3;
% dx(2,n+1)=dx(2,n)+(A(2)+2*B(2)+2*C(2)+D(2))/3;
% time(n+1)=(n+1)*dt;
% 
% end;
% 
% 
% plot(time,x(1,:),'.g','Linewidth',2); 
% hold on
% plot(time,x(2,:),'.c','Linewidth',2);
% % legend ('x','\theta');
% xlabel('time [s]','FontSize',15)
% ylabel('x,\theta','FontSize',20)
% set(gca,'FontSize',15)
% % grid
