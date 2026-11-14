"""CUAD categories, prompts and scoring.

The categories, the label construction and the span-matching rule (Jaccard >= 0.15, substring
match for "Parties") follow Palimpzest's abacus-research/cuad-demo.py (MIT license), so F1 is
comparable with the Abacus paper. Contracts are split into chunks; predictions from all chunks
of a contract are merged before scoring.
"""
import json
import re
import string

CATEGORIES = [
    ("Document Name", "The name of the contract"),
    ("Parties", "The two or more parties who signed the contract"),
    ("Agreement Date", "The date of the contract"),
    ("Effective Date", "The date when the contract is effective"),
    ("Expiration Date", "On what date will the contract's initial term expire?"),
    ("Renewal Term", "What is the renewal term after the initial term expires? This includes automatic extensions and unilateral extensions with prior notice."),
    ("Notice Period to Terminate Renewal", "What is the notice period required to terminate renewal?"),
    ("Governing Law", "Which state/country's law governs the interpretation of the contract?"),
    ("Most Favored Nation", "Is there a clause that if a third party gets better terms on the licensing or sale of technology/goods/services described in the contract, the buyer of such technology/goods/services under the contract shall be entitled to those better terms?"),
    ("Non-Compete", "Is there a restriction on the ability of a party to compete with the counterparty or operate in a certain geography or business or technology sector?"),
    ("Exclusivity", "Is there an exclusive dealing  commitment with the counterparty? This includes a commitment to procure all “requirements” from one party of certain technology, goods, or services or a prohibition on licensing or selling technology, goods or services to third parties, or a prohibition on  collaborating or working with other parties), whether during the contract or  after the contract ends (or both)."),
    ("No-Solicit of Customers", "Is a party restricted from contracting or soliciting customers or partners of the counterparty, whether during the contract or after the contract ends (or both)?"),
    ("Competitive Restriction Exception", "This category includes the exceptions or carveouts to Non-Compete, Exclusivity and No-Solicit of Customers above."),
    ("No-Solicit of Employees", "Is there a restriction on a party’s soliciting or hiring employees and/or contractors from the  counterparty, whether during the contract or after the contract ends (or both)?"),
    ("Non-Disparagement", "Is there a requirement on a party not to disparage the counterparty?"),
    ("Termination for Convenience", "Can a party terminate this  contract without cause (solely by giving a notice and allowing a waiting  period to expire)?"),
    ("Rofr/Rofo/Rofn", "Is there a clause granting one party a right of first refusal, right of first offer or right of first negotiation to purchase, license, market, or distribute equity interest, technology, assets, products or services?"),
    ("Change of Control", "Does one party have the right to terminate or is consent or notice required of the counterparty if such party undergoes a change of control, such as a merger, stock sale, transfer of all or substantially all of its assets or business, or assignment by operation of law?"),
    ("Anti-Assignment", "Is consent or notice required of a party if the contract is assigned to a third party?"),
    ("Revenue/Profit Sharing", "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?"),
    ("Price Restrictions", "Is there a restriction on the  ability of a party to raise or reduce prices of technology, goods, or  services provided?"),
    ("Minimum Commitment", "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?"),
    ("Volume Restriction", "Is there a fee increase or consent requirement, etc. if one party’s use of the product/services exceeds certain threshold?"),
    ("IP Ownership Assignment", "Does intellectual property created  by one party become the property of the counterparty, either per the terms of the contract or upon the occurrence of certain events?"),
    ("Joint IP Ownership", "Is there any clause providing for joint or shared ownership of intellectual property between the parties to the contract?"),
    ("License Grant", "Does the contract contain a license granted by one party to its counterparty?"),
    ("Non-Transferable License", "Does the contract limit the ability of a party to transfer the license being granted to a third party?"),
    ("Affiliate License-Licensor", "Does the contract contain a license grant by affiliates of the licensor or that includes intellectual property of affiliates of the licensor?"),
    ("Affiliate License-Licensee", "Does the contract contain a license grant to a licensee (incl. sublicensor) and the affiliates of such licensee/sublicensor?"),
    ("Unlimited/All-You-Can-Eat-License", "Is there a clause granting one party an “enterprise,” “all you can eat” or unlimited usage license?"),
    ("Irrevocable or Perpetual License", "Does the contract contain a  license grant that is irrevocable or perpetual?"),
    ("Source Code Escrow", "Is one party required to deposit its source code into escrow with a third party, which can be released to the counterparty upon the occurrence of certain events (bankruptcy,  insolvency, etc.)?"),
    ("Post-Termination Services", "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?"),
    ("Audit Rights", "Does a party have the right to  audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?"),
    ("Uncapped Liability", "Is a party’s liability uncapped upon the breach of its obligation in the contract? This also includes uncap liability for a particular type of breach such as IP infringement or breach of confidentiality obligation."),
    ("Cap on Liability", "Does the contract include a cap on liability upon the breach of a party’s obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery."),
    ("Liquidated Damages", "Does the contract contain a clause that would award either party liquidated damages for breach or a fee upon the termination of a contract (termination fee)?"),
    ("Warranty Duration", "What is the duration of any  warranty against defects or errors in technology, products, or services  provided under the contract?"),
    ("Insurance", "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?"),
    ("Covenant Not to Sue", "Is a party restricted from contesting the validity of the counterparty’s ownership of intellectual property or otherwise bringing a claim against the counterparty for matters unrelated to the contract?"),
    ("Third Party Beneficiary", "Is there a non-contracting party who is a beneficiary to some or all of the clauses in the contract and therefore can enforce its rights against a contracting party?"),
]
NAMES = [name for name, _ in CATEGORIES]
IOU_THRESH = 0.15


def label_category(qa_id):
    """Category name from a CUAD question id, normalised as in cuad-demo.py."""
    name = qa_id.split("__")[-1].split("_")[0].strip()
    for a, b in ((" For ", " for "), (" Of ", " of "), (" On ", " on "), (" Or ", " or "), (" To ", " to "), ("Ip", "IP")):
        name = name.replace(a, b)
    return name


def parse_answer(text, names):
    """Best-effort parse of a model answer into category -> list of spans. Returns (dict, ok)."""
    out = {name: [] for name in names}
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return out, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out, False
    if not isinstance(obj, dict):
        return out, False
    for name in names:
        val = obj.get(name)
        if isinstance(val, str):
            val = [val]
        if isinstance(val, list):
            out[name] = [str(v).strip() for v in val if str(v).strip()]
    return out, True


def _jaccard(label, pred):
    remove = [c for c in string.punctuation if c != "/"]
    for c in remove:
        label = label.replace(c, "")
        pred = pred.replace(c, "")
    label_words = set(label.lower().replace("/", " ").split(" "))
    pred_words = set(pred.lower().replace("/", " ").split(" "))
    return len(label_words & pred_words) / len(label_words | pred_words)


def _match(label, pred, substr_ok):
    return _jaccard(label, pred) >= IOU_THRESH or (substr_ok and label in pred)


def _entry(labels, preds, substr_ok):
    if not labels:
        return 0, len(preds), 0
    tp = sum(any(_match(a, p, substr_ok) for p in preds) for a in labels)
    fn = len(labels) - tp
    fp = sum(not any(_match(a, p, substr_ok) for a in labels) for p in preds)
    return tp, fp, fn


def score(labels, preds):
    """labels, preds: contract_id -> category -> list of spans. Micro precision, recall and F1."""
    tp = fp = fn = 0
    for cid, cat_labels in labels.items():
        cat_preds = preds.get(cid, {})
        for name in NAMES:
            t, f_p, f_n = _entry(cat_labels.get(name, []), cat_preds.get(name, []), "Parties" in name)
            tp, fp, fn = tp + t, fp + f_p, fn + f_n
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def merge_chunks(records, states, keys):
    """Union of the spans each chunk's answers give, per contract. Also counts unparsable answers."""
    preds, failures = {}, 0
    for r in records:
        merged = preds.setdefault(r["contract_id"], {name: [] for name in NAMES})
        for key in keys:
            names = r.get("categories", {}).get(key) or NAMES
            parsed, ok = parse_answer(states[r["id"]].get(key, ""), names)
            failures += not ok
            for name, spans in parsed.items():
                merged[name].extend(s for s in spans if s not in merged[name])
    return preds, failures
