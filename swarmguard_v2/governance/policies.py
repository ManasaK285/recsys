class Policy:
    def __init__(self,name,audit=False,quarantine=False,recovery=False,whistleblowing=True):
        self.name=name; self.audit=audit; self.quarantine=quarantine; self.recovery=recovery; self.whistleblowing=whistleblowing
POLICIES={
 "baseline":Policy("baseline"),
 "peer_audit":Policy("peer_audit",audit=True),
 "audit_quarantine":Policy("audit_quarantine",audit=True,quarantine=True),
 "audit_quarantine_recovery":Policy("audit_quarantine_recovery",audit=True,quarantine=True,recovery=True),
}
