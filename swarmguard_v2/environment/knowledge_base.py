class KnowledgeBase:
    def __init__(self):
        self.entries=[]
    def publish(self, author, kind, content, round_idx):
        self.entries.append({"author":author,"kind":kind,"content":content,"round":round_idx})
    def exploit_visible(self):
        return any(e["kind"]=="EXPLOIT" for e in self.entries)
    def alerts(self):
        return [e for e in self.entries if e["kind"]=="ALERT"]
