"""Refresh existing ten-case HTML with shared G3 CSS; no image reads or writes."""
import argparse
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
from urllib.parse import urlsplit,unquote
from agent_baseline.make_demo import CSS,CURRENT_CSS
from .io_retry import read_text,write_text,write_json,exists
from .t3_routes import pooled


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set();self.images=0
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='img':self.images+=1
        for key in ('href','src'):
            if key in a:self.links.append(a[key])


def refresh(out):
    root=out/'diagnostics10';manifest=json.loads(read_text(root/'manifest.json'))
    advancement=json.loads(read_text(out/'advancement.json'))
    extra='''
    .hub{max-width:1500px;margin:auto;padding:16px}.n8-card{margin:18px 0;overflow:hidden}
    .n8-card img{display:block;width:100%;height:auto;max-width:100%;border:1px solid var(--line)}
    .n8-links{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0}.n8-meta{display:flex;flex-wrap:wrap;gap:8px}
    .n8-chip{background:var(--tint);padding:5px 9px;border-radius:4px}.n8-candidates{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:8px}
    .n8-card table{width:100%}.n8-note{color:var(--muted)}.n8-legend span{margin-right:16px}
    @media(max-width:700px){.hub{padding:8px}.n8-card{padding:10px}.n8-links{gap:8px}}
    '''
    lines=['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>N8 T3 · 十例完整诊断</title><style>'+CSS+CURRENT_CSS+extra+'</style></head><body>',
        '<header><h1>N8 T3 · 确认集十例诊断</h1><p>沿用 G3 页面视觉样式；本页是离线诊断，不是在线 Agent 回放。图像、示例、预测和 GT 均保留完整原生视野。</p>',
        '<nav class="n8-links" aria-label="研究证据"><a href="../research_review.md">专项报告</a><a href="../predeclaration.md">预声明</a><a href="../step3/report.md">完整指标</a><a href="../final_audit.json">逐例损失与费用</a><a href="manifest.json">十例选样记录</a></nav>',
        f'<div class="n8-meta"><span class="n8-chip">最终非空 F1 {advancement["positive_N8"]["f1_c"]:.3f}</span><span class="n8-chip">基线 {advancement["positive_baseline"]["f1_c"]:.3f}</span><span class="n8-chip">空 FP {advancement["empty_N8"]["fp"]}</span><span class="n8-chip">门控主方案未晋级；Top5对照另行在线验证</span></div>',
        '<p class="n8-legend"><span style="color:#1cbe55">绿：TP</span><span style="color:#f43fa0">洋红：FP</span><span style="color:#00a9d6">青：FN</span><span style="color:#b29000">黄：示例</span></p>',
        '<p class="n8-note">含两例 tiny-GT、清理自匹配收益、最终损害/收益及空目标误检；选样和目视只用于解释，不改变 GT、参数或主配置。</p></header><main class="hub">']
    for number,item in enumerate(manifest['cases'],1):
        ident=item['id'];case=json.loads(read_text(root/ident/'case.json'));m=case['metrics'];esc=html.escape
        pair=case['observable']['predicted_pair'];channels=case['observable']['channels']
        labels=[channels[i] for i in pair]
        lines += [f'<section class="panel n8-card" id="case-{number}"><h2>{number:02d} · {esc(ident)}</h2>',
            f'<p>{esc(item["reason"])}</p><div class="n8-meta"><span class="n8-chip">剩余 GT {m["remaining_gt_pixels"]} px</span><span class="n8-chip">E 预测 {esc(labels[0])} → {esc(labels[1])}</span><span class="n8-chip">gate {m["triggered"]} · z {m["z"]:.3f} · p {m["top1_ptrans"]:.3f}</span></div>',
            '<div class="tablewrap"><table><thead><tr><th>阶段</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th></tr></thead><tbody>']
        for name,label in [('original_R1_top8','原 R1 Top8'),('N8_gate_unverified','冻结 gate/K · 未核验'),('N8_verified','最终 N8')]:
            value=pooled([m['methods'][name]])
            lines.append(f'<tr><td>{label}</td><td>{value["f1_c"]:.3f}</td><td>{value["tp"]}</td><td>{value["fp"]}</td><td>{value["fn"]}</td></tr>')
        lines+=['</tbody></table></div><div class="n8-links">']
        for filename,label in [('T1.png','原始T1'),('T2.png','原始T2'),('example.png','示例E'),('GT_remaining.png','剩余GT'),('N8_prediction.png','最终掩码'),('case.json','候选/答案/概率JSON')]:
            lines.append(f'<a href="{esc(ident)}/{filename}">{label}</a>')
        lines += ['</div>',f'<a href="{esc(ident)}/panel.png"><img loading="lazy" src="{esc(ident)}/panel.png" alt="{esc(ident)} 的原生完整双期、示例、各阶段预测与GT面板"></a>',
                  '<details><summary>前五候选：两期原图覆盖、掩码和实际回答</summary><div class="n8-candidates">']
        for c in case['candidates']:
            i=c['rank'];answer=(c.get('answer') or {}).get('answer','未送审')
            lines.append(f'<div class="flowbox"><strong>候选{i} · {esc(str(answer))}</strong><p>面积 {c["area"]} · 相似度 {c["similarity"]:.3f} · ptrans {c["ptrans"]:.3f}</p><div class="n8-links"><a href="{esc(ident)}/candidate_{i}_T1.png">T1</a><a href="{esc(ident)}/candidate_{i}_T2.png">T2</a><a href="{esc(ident)}/candidate_{i}_mask.png">掩码</a></div></div>')
        lines.append('</div></details></section>')
    lines.append('</main></body></html>');document='\n'.join(lines)
    old=root/'index.html'
    if exists(old) and not exists(root/'index_unstyled.html'):write_text(root/'index_unstyled.html',read_text(old))
    write_text(old,document)
    parser=Links();parser.feed(document);missing=[]
    for link in parser.links:
        target=urlsplit(link)
        if target.scheme or target.netloc:continue
        if not target.path:
            if target.fragment and target.fragment not in parser.ids:missing.append(link)
        elif not exists(root/unquote(target.path)):missing.append(link)
    audit=dict(cases=len(manifest['cases']),images=parser.images,relative_links_checked=len(parser.links),missing=missing,
        style_source='agent_baseline.make_demo.CSS + CURRENT_CSS',image_policy='HTML/JSON only; original PNGs neither opened nor regenerated',status='complete' if not missing else 'broken_links')
    write_json(root/'html_audit.json',audit)
    if missing:raise ValueError('Missing relative links: '+', '.join(missing))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm for report rendering/checks')
    refresh(a.out)


if __name__=='__main__':main()
