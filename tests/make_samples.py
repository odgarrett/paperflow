"""Build synthetic alert emails that mimic each source's layout."""
from email.message import EmailMessage
from pathlib import Path

out = Path(__file__).parent / "samples"

def save(name, sender, subject, html=None, text=None, date="Mon, 06 May 2024 09:00:00 +0000"):
    m = EmailMessage()
    m["From"], m["To"], m["Subject"], m["Date"] = sender, "me@uni.edu", subject, date
    m["Message-ID"] = f"<{name}@test>"
    if text: m.set_content(text)
    if html:
        if text: m.add_alternative(html, subtype="html")
        else: m.set_content(html, subtype="html")
    (out / f"{name}.eml").write_bytes(bytes(m))

def scholar_item(url, title, byline, snippet, folt="kw", pdf=False):
    wrapped = ("https://scholar.google.com/scholar_url?url=" + url.replace("&", "%26") +
               f"&hl=en&sa=X&d=123&ei=abc&scisig=xyz&oi=scholaralrt&hist=h&html=&pos=0&folt={folt}")
    tag = '<span style="font-size:11px">[PDF]</span> ' if pdf else ""
    return (f'<h3 style="font-weight:normal">{tag}<a href="{wrapped}" class="gse_alrt_title">{title}</a></h3>'
            f'<div style="color:#006621;line-height:18px">{byline}</div>'
            f'<div class="gse_alrt_sni" style="line-height:17px">{snippet}</div>'
            '<div style="width:auto"><table><tr><td><a href="https://scholar.google.com/scholar_share?x=1">Save</a></td>'
            '<td><a href="https://twitter.com/intent/tweet?x">Twitter</a></td></tr></table></div><br>')

html = ("<html><body>" +
    scholar_item("https://www.biorxiv.org/content/10.1101/2024.05.01.590001v1.abstract",
                 "Dopamine neurons encode prediction errors across the whole…",
                 "A Garcia, B Chen, C Okafor… - bioRxiv, 2024",
                 "… we recorded from midbrain <b>dopamine</b> neurons …") +
    scholar_item("https://www.sciencedirect.com/science/article/pii/S0896627324001234",
                 "Thalamic gating of cortical plasticity during sleep",
                 "K Tanaka, M Rossi - Neuron, 2024",
                 "… sleep spindles coordinate …", pdf=True) +
    scholar_item("https://arxiv.org/pdf/2405.01234",
                 "A normative model of exploration in foraging animals",
                 "J Park, L Silva - arXiv preprint arXiv:2405.01234, 2024",
                 "… we propose a normative …") +
    '<p><a href="https://scholar.google.com/scholar_alerts?view_op=list_alerts">List alerts</a> '
    '<a href="https://scholar.google.com/scholar_alerts?view_op=cancel_alert_options">Cancel alert</a></p>'
    "</body></html>")
save("scholar_kw", "Google Scholar Alerts <scholaralerts-noreply@google.com>",
     "dopamine prediction error - new results", html)

html2 = ("<html><body>" +
    scholar_item("https://www.sciencedirect.com/science/article/pii/S0896627324001234",
                 "Thalamic gating of cortical plasticity during sleep",
                 "K Tanaka, M Rossi - Neuron, 2024", "… spindles …", folt="cit") +
    "</body></html>")
save("scholar_cit", "Google Scholar Alerts <scholaralerts-noreply@google.com>",
     "2 new citations to articles by Jane Doe", html2, date="Tue, 07 May 2024 09:00:00 +0000")

bx = """<html><body><h2>bioRxiv Neuroscience alert</h2>
<table>
<tr><td><a href="https://www.biorxiv.org/content/10.1101/2024.05.01.590001v1?rss=1">Dopamine neurons encode prediction errors across the whole striatum</a><br>
Garcia, A.; Chen, B.; Okafor, C.<br>Posted May 03, 2024</td></tr>
<tr><td><a href="https://www.biorxiv.org/content/10.1101/2024.05.02.590777v1?rss=1">Astrocytes regulate synaptic scaling in visual cortex</a><br>
Ng, P.; Weber, S.<br><a href="https://www.biorxiv.org/content/10.1101/2024.05.02.590777v1.full.pdf">PDF</a></td></tr>
</table>
<p><a href="https://connect.biorxiv.org/alerts/manage">Manage alerts</a> | <a href="https://connect.biorxiv.org/unsubscribe?x=1">Unsubscribe</a></p>
</body></html>"""
save("biorxiv", "bioRxiv <cshljnls-mailer@alerts.highwire.org>", "bioRxiv Subject Collection Alert: Neuroscience", bx,
     date="Sat, 04 May 2024 09:00:00 +0000")

stork = """<html><body><p>Stork found 2 new papers for your keyword: dopamine</p>
<div><a href="https://www.storkapp.me/paper/showPaper.php?id=111"><b>Dopamine neurons encode prediction errors across the whole striatum</b></a>
<br>Nature Neuroscience | Garcia A, Chen B, Okafor C | <a href="https://pubmed.ncbi.nlm.nih.gov/38000001/">PubMed</a></div>
<div><a href="https://www.storkapp.me/paper/showPaper.php?id=112"><b>Cholinergic interneurons pause during reward omission in mice</b></a>
<br>Journal of Neuroscience | Ito R, Banerjee S</div>
<p><a href="https://www.storkapp.me/settings">Settings</a></p></body></html>"""
save("stork", "Stork <noreply@storkapp.me>", "Stork: new papers for dopamine", stork,
     date="Wed, 15 May 2024 09:00:00 +0000")

text = """New articles in your alert:

Cholinergic interneurons pause during reward omission in mice
https://doi.org/10.1523/JNEUROSCI.0001-24.2024

Unsubscribe: https://example.org/unsubscribe
"""
save("plaintext", "Journal TOC <toc@jneurosci.org>", "J Neurosci alert", text=text,
     date="Thu, 16 May 2024 09:00:00 +0000")
print("samples written:", sorted(p.name for p in out.iterdir()))
