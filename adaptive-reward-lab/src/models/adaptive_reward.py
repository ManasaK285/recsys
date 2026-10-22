import numpy as np

class AdaptiveReward:
    """Learn shared nonlinear features and a user-specific linear reward vector."""
    def __init__(self, input_dim, feature_dim=8, n_users=80, lr=.02, epochs=300, l2=1e-3, seed=42):
        self.input_dim, self.feature_dim, self.n_users = input_dim, feature_dim, n_users
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0,.25,(input_dim,feature_dim))
        self.b = np.zeros(feature_dim)
        self.V = rng.normal(0,.1,(n_users,feature_dim))

    @staticmethod
    def sigmoid(z):
        return 1/(1+np.exp(-np.clip(z,-30,30)))

    def phi(self, x):
        return np.tanh(np.asarray(x) @ self.W + self.b)

    def fit(self, a, b, y, user_ids):
        a,b,y,user_ids = np.asarray(a),np.asarray(b),np.asarray(y),np.asarray(user_ids,dtype=int)
        n=max(len(y),1)
        for _ in range(self.epochs):
            pa,pb=self.phi(a),self.phi(b)
            delta=pa-pb
            err=(self.sigmoid(np.sum(delta*self.V[user_ids],axis=1))-y)/n
            gv=np.zeros_like(self.V); np.add.at(gv,user_ids,err[:,None]*delta)
            gd=err[:,None]*self.V[user_ids]
            ga=gd*(1-pa**2); gb=-gd*(1-pb**2)
            gw=(a.T@ga+b.T@gb)/n+self.l2*self.W
            gbias=(ga.sum(axis=0)+gb.sum(axis=0))/n
            self.V-=self.lr*(gv+self.l2*self.V/max(self.n_users,1))
            self.W-=self.lr*gw; self.b-=self.lr*gbias
        return self

    def predict_proba(self,a,b,user_ids):
        ids=np.asarray(user_ids,dtype=int)
        ids=np.clip(ids,0,self.n_users-1)
        return self.sigmoid(np.sum((self.phi(a)-self.phi(b))*self.V[ids],axis=1))

    def adapt_user(self,a,b,y,steps=150,lr=.05,l2=.01):
        """Fit new-user weights with shared feature parameters frozen."""
        d=self.phi(a)-self.phi(b); y=np.asarray(y,dtype=float); v=np.zeros(self.feature_dim)
        for _ in range(steps):
            p=self.sigmoid(d@v)
            v-=lr*(d.T@(p-y)/max(len(y),1)+l2*v)
        return v

    def predict_with_weights(self,a,b,weights):
        return self.sigmoid(np.sum((self.phi(a)-self.phi(b))*weights,axis=1))
