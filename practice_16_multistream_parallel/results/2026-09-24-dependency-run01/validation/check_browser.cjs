const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});const errors=[],requests=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///home/tianchi/workspace/AI-Serving-Infra-Study/practice_16_multistream_parallel/results/2026-09-24-dependency-run01/analysis/index.html');
 assert.equal(await page.locator('.trial:visible').count(),6);assert.equal(await page.locator('.task').count(),99);
 assert.equal(await page.locator('[data-label$="/wait-A"]').count(),3);
 for(const value of await page.locator('#trial option').evaluateAll(xs=>xs.map(x=>x.value))){
  await page.locator('#trial').selectOption(value);assert.equal(await page.locator('.trial:visible').count(),value==='all'?6:1);
  await page.locator('.trial:visible .task').first().click();const d=JSON.parse(await page.locator('#detail').textContent());assert(d.torch_flow&&d.cann_flow);
 }
 await page.locator('#trial').selectOption('trace-00-event_wait');
 await page.locator('[data-label="P16Dep/trace-00-event_wait/wait-A"]').focus();await page.keyboard.press('Enter');
 assert.equal(JSON.parse(await page.locator('#detail').textContent()).kind,'device_wait');
 assert((await page.locator('body').textContent()).includes('3 / 3 个保留依赖'));
 for(const url of await page.locator('a').evaluateAll(as=>as.map(a=>a.href)))assert(fs.existsSync(new URL(url)),url);
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'/tmp/p16-dependency-safe.png',fullPage:true});
 await page.locator('#trial').selectOption('trace-00-no_wait');await page.screenshot({path:'/tmp/p16-dependency-unsafe.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.screenshot({path:'/tmp/p16-dependency-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
 console.log(JSON.stringify({offline:true,trial_filters:7,device_tasks:99,event_wait_tasks:3,click_evidence:true,keyboard:true,links:true,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
