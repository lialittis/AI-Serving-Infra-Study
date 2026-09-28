const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');const assert=require('assert'),fs=require('fs');
const root='/home/tianchi/workspace/AI-Serving-Infra-Study/practice_15_kernel_execution_graph/results/';
(async()=>{const b=await chromium.launch({headless:true});try{const page=await b.newPage({viewport:{width:1900,height:1200},offline:true});const errors=[];page.on('pageerror',e=>errors.push(e.message));
for(const mode of ['eager','graph']){
 await page.goto('file://'+root+'2026-09-28-'+mode+'-run01/analysis/index.html');
 assert.equal(await page.locator('nav a').count(),4);
 for(const link of await page.locator('nav a').evaluateAll(xs=>xs.map(x=>x.href))){assert(fs.existsSync(new URL(link)));const p=await b.newPage();await p.goto(link);assert(await p.locator('svg .node').count()>0);assert(await p.locator('svg .edge').count()>0);await p.close()}
}
for(const phase of ['prefill','decode-1']){
 await page.goto('file://'+root+'2026-09-28-graph-run01/analysis/first_attention_'+phase+'.svg');
 await page.locator('svg').evaluate(e=>{e.style.width='1850px';e.style.height=(1850*e.viewBox.baseVal.height/e.viewBox.baseVal.width)+'px'});
 await page.screenshot({path:'/tmp/p15-focus-'+phase+'.png',fullPage:false});
}
assert.deepEqual(errors,[]);console.log(JSON.stringify({offline:true,linked_svgs:8,rendered:true,errors}));
}finally{await b.close()}})().catch(e=>{console.error(e);process.exit(1)});
