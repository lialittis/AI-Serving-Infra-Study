const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});const errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///home/tianchi/workspace/AI-Serving-Infra-Study/practice_16_multistream_parallel/results/2026-09-24-sync-run01/analysis/index.html');
 const options=await page.locator('#sample option').evaluateAll(xs=>xs.map(x=>x.value));assert.equal(options.length,14);
 for(const value of options){await page.locator('#sample').selectOption(value);assert((await page.locator('#timing').textContent()).includes('完整耗时'));}
 assert((await page.locator('body').textContent()).includes('56 次分支采样'));
 await page.locator('#sample').selectOption('bench-00-final_only');
 assert((await page.locator('#timing').textContent()).includes('提交阶段 1.199 ms'));
 assert.equal(await page.locator('tbody tr').count(),10);
 for(const url of await page.locator('a').evaluateAll(as=>as.map(a=>a.href)))assert(fs.existsSync(new URL(url)),url);
 await page.screenshot({path:'/tmp/p16-sync-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.screenshot({path:'/tmp/p16-sync-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
 console.log(JSON.stringify({offline:true,sample_options:14,timing_bars:true,early_read_table:true,links:true,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
