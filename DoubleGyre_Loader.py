import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class DG_Dataset(torch.utils.data.Dataset):
    d_measure = 1

    def __init__(self, size, seed=None, 
                 seq_t=4, dt=0.01, transform=None):
        self.transform = transform

        # set random generator
        rand = np.random.default_rng(seed)

        self.total_t = 5
        self.seq_t = seq_t
        self.dt = dt
        self.total_length = int(self.total_t/self.dt)
        self.seq_length = int(self.seq_t/self.dt)
        self.size = size
        
        self.A = 0.5
        self.omega = 2*np.pi
        self.eps = 0.25
        self.width=201
        self.height=101

        # generate full date for time T=[0,5] with dt
        T = np.arange(0,self.total_t+self.dt,self.dt)
        assert len(T) == self.total_length+1
        X,Y = np.meshgrid(np.linspace(0,2,self.width),np.linspace(0,1,self.height))
        self.data = np.array([self._curl(X,Y,t).flatten() for t in T])

        # generate random trajectories
        self.time = rand.choice(len(T)-self.seq_length,size=self.size) # get random initial time
        self.traj = np.zeros((len(self.time), self.seq_length), dtype=int)
        i = 0
        while i < len(self.time):
            t = self.time[i] * self.dt
            init_y = np.round(rand.uniform((0.04,0.04),(1.96,0.96)),2) # random initial position
            sol = solve_ivp(lambda t, y: self._get_v(t,y), (t, t+self.seq_t+0.001), init_y, t_eval = np.arange(self.seq_length)*self.dt+t)
            # convert traj to index
            a = np.round(sol.y,2)
            if np.any(a[0] < 0) or np.any(a[0] > 2) or np.any(a[1] < 0) or np.any(a[1] > 1):
                continue
            # assert np.all(a[0] >= 0) and np.all(a[0] <= 2) and np.all(a[1] >= 0) and np.all(a[1] <= 1), a
            self.traj[i] = (a[0]/0.01) + (a[1]/0.01)*self.width
            i += 1

    # double gyre functions
    def _f(self,x,t):
        return self.eps*np.sin(self.omega*t)*x**2 + x - 2*self.eps*np.sin(self.omega*t)*x
    def _df(self,x,t):
        return 2*self.eps*np.sin(self.omega*t)*x + 1 - 2*self.eps*np.sin(self.omega*t)
    def _ddf(self,x,t):
        return 2*self.eps*np.sin(self.omega*t)
    # def _U(self,x,y,t):
    #     return -np.pi*self.A*np.sin(np.pi*self._f(x,t))*np.cos(np.pi*y)
    # def _V(self,x,y,t):
    #     return np.pi*self.A*np.cos(np.pi*self._f(x,t))*np.sin(np.pi*y)*self._df(x,t)
    def _get_v(self,t,loc):
        x,y = loc
        u = -np.pi*self.A*np.sin(np.pi*self._f(x,t))*np.cos(np.pi*y) #self._U(x,y,t) #
        v = np.pi*self.A*np.cos(np.pi*self._f(x,t))*np.sin(np.pi*y)*self._df(x,t) #self._V(x,y,t) #
        return np.array([u,v])
    def _curl(self,x,y,t):
        return -np.pi**2*self.A*np.sin(np.pi*self._f(x,t))*np.sin(np.pi*y)*(1+self._df(x,t)**2) + \
            np.pi*self.A*np.cos(np.pi*self._f(x,t))*np.sin(np.pi*y)*self._ddf(x,t)
    

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        t = self.time[index]
        traj = self.traj[index]
        sample = np.array([self.data[t+i,traj[i]] for i in range(self.seq_length)])
        sample = np.array([sample, traj/(self.width*self.height)]).T
        if self.transform:
            sample = self.transform(sample)
        return (
            torch.tensor(sample,dtype=torch.float32),
            torch.tensor(self.data[t:t+self.seq_length],dtype=torch.float32),
        )

    # plot
    def plot_traj(self,input,t=None,c='k'):
        input = input * self.width * self.height
        x,y = input%self.width, input//self.width
        plt.plot(x,y,color=c)
        if t is not None:
            plt.scatter(x[t],y[t],color='r')

    def plot_data(self,data,cbar=False,err_cm=False):
        if not err_cm:
            plt.imshow(data.reshape(self.height,self.width),origin='lower')
        else:
            plt.imshow(data.reshape(self.height,self.width), cmap=plt.cm.cividis,origin='lower')
        # plt.imshow(data.reshape(self.height,self.width))
        plt.xticks([])
        plt.yticks([])
        if cbar:
            im_ratio = self.height/self.width
            plt.colorbar(fraction=0.047*im_ratio)
            # plt.colorbar(fig,fraction=0.047*im_ratio)