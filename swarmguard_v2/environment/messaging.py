class MessageBus:
    def __init__(self): self.messages=[]
    def broadcast(self, sender, kind, content, round_idx):
        self.messages.append({"sender":sender,"kind":kind,"content":content,"round":round_idx,"broadcast":True})
    def direct(self, sender, receiver, kind, content, round_idx):
        self.messages.append({"sender":sender,"receiver":receiver,"kind":kind,"content":content,"round":round_idx,"broadcast":False})
