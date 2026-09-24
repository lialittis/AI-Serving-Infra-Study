const {chromium}=require('/tmp/inference-report-check/node_modules/playwright');
const assert=require('assert'),fs=require('fs');
(async()=>{const browser=await chromium.launch({headless:true});try{
 const page=await browser.newPage({viewport:{width:1440,height:1100},offline:true});
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.goto('file:///home/tianchi/workspace/AI-Serving-Infra-Study/practice_16_multistream_parallel/results/2026-09-24-run02/analysis/index.html');
 assert.equal(await page.locator('.trial:visible').count(),6);
 assert.equal(await page.locator('.kernel').count(),72);
 for(const value of await page.locator('#trial option').evaluateAll(xs=>xs.map(x=>x.value))){
  await page.locator('#trial').selectOption(value);
  assert.equal(await page.locator('.trial:visible').count(),value==='all'?6:1);
  await page.locator('.trial:visible .kernel').first().click();
  const data=JSON.parse(await page.locator('#detail').textContent());assert(data.torch_flow&&data.cann_flow&&data.stream);
 }
 await page.locator('#trial').selectOption('01-parallel');
 await page.locator('.kernel[data-label="P16/01-parallel/mul-05"]').focus();await page.keyboard.press('Enter');
 assert((await page.locator('#detail').textContent()).includes('mul-05'));
 for(const url of await page.locator('a').evaluateAll(as=>as.map(a=>a.href))) assert(fs.existsSync(new URL(url)),url);
 await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:'/tmp/p16-desktop.png',fullPage:true});
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
 await page.screenshot({path:'/tmp/p16-mobile.png',fullPage:true});
 assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
 console.log(JSON.stringify({offline:true,trial_filters:7,kernel_bars:72,click_evidence:true,keyboard:true,links:true,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
