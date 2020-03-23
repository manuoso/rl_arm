%% POSITRON KINEMATICS ANALYSIS
% Using Peter Corke Toolbox

clc; clear all;

L1 = 10.0;
L2 = 200.0;
L3 = 200.0;

L(1)= Link([0 L1 0 pi/2]);
L(2)= Link([0 0 L2 0]);
L(3)= Link([0 0 L3 0]);

% set limits for joints
% L(1).qlim = [deg2rad(-150) deg2rad(150)];
% L(2).qlim = [deg2rad(-90) deg2rad(90)];
% L(3).qlim = [deg2rad(-120) deg2rad(120)];

robot = SerialLink(L);
robot.name = 'hecatonquiros';

% figure
% hold on
% view(3)
% grid on
% 
% robot.teach();
% robot.plot([pi/2 pi/2 pi/2])

% for i=-150:5:150
%     for j=-90:5:90
%         for k=-120:5:120
%             Mricx = robot.fkine([i*pi/180 j*pi/180 k*pi/180]);
%             hold on
%             plot3(Mricx.t(1),Mricx.t(2),Mricx.t(3),'b.','MarkerSize',0.5);
%             pause(0.001)
%         end
%     end
% end

[~,n] = size(L);

var = sym('q',[n 1]);
assume(var,'real')

% generate a grid of theta1 and theta2,3,4 values

th1 = (-150:5:150);
th2 = (-90:5:90);
th3 = (-120:5:120);

q = {th1*pi/180,th2*pi/180,th3*pi/180};

[Q{1:numel(q)}] = ndgrid(q{:}); 
T = simplify(vpa(robot.fkine(var),3));
Pos = T.tv;

x(var(:)) = Pos(1);
X = matlabFunction(x);
X = X(Q{:});

y(var(:)) = Pos(2);
Y = matlabFunction(y);
Y = Y(Q{:});

z(var(:)) = Pos(3);
Z = matlabFunction(z);
Z = Z(Q{:});

plot3(X(:),Y(:),Z(:),'b.')
xlabel('X')
ylabel('Y')
zlabel('Z')
grid on

