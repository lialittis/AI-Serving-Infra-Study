const {chromium}=require(process.env.P23_BROWSER_PACKAGE||'playwright');
const assert=require('assert'),{pathToFileURL}=require('url'),path=require('path');
(async()=>{const browser=await chromium.launch({headless:true});try{
const page=await browser.newPage({viewport:{width:1400,height:1000},offline:true});
const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
await page.goto(pathToFileURL(path.resolve(__dirname,'report/index.html')).href);await page.locator('#case option').first().waitFor({state:'attached'});
let trials=0;
for(const c of ['prefill-128','prefill-1024','decode-128','decode-1024'])for(const mode of ['serial','parallel','batch'])for(const r of ['0','1']){
 await page.locator('#case').selectOption(c);await page.locator('#mode').selectOption(mode);await page.locator('#repeat').selectOption(r);
 const tasks=page.locator('[data-node]');assert(await tasks.count()>1000);
 const index=await tasks.evaluateAll(xs=>xs.findIndex(e=>e.tagName==='rect'&&Number(e.getAttribute('width'))>2));assert(index>=0);
 // A real click on a sufficiently wide kernel, including scrolling into view.
 await tasks.nth(index).click();const details=JSON.parse(await page.locator('#details').textContent());assert(details.node.cann_flow);assert(details.node.csv_row!==undefined);
 assert.equal((await page.locator('#stats').textContent()).includes('数值未通过'),c==='prefill-1024'&&mode==='batch');trials++;
}
for(const kind of ['event_wait','stream_order',''])await page.locator('#edges').selectOption(kind);
await page.locator('#case').selectOption('prefill-1024');await page.locator('#mode').selectOption('parallel');await page.locator('#repeat').selectOption('0');
assert((await page.locator('#details').textContent()).includes('点击当前实验'));
await page.locator('#edges').selectOption('event_wait');await page.locator('#zoom').selectOption('10');await page.locator('#zoom').selectOption('1');
await page.screenshot({path:'/tmp/p23-desktop.png',fullPage:true});await page.setViewportSize({width:390,height:844});
assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));await page.screenshot({path:'/tmp/p23-mobile.png',fullPage:true});
assert.deepEqual(errors,[]);assert(requests.every(r=>r.startsWith('file:')));
console.log(JSON.stringify({offline:true,trials,kernel_details:true,invalid_batch_visible:true,edge_filters:3,mobile_no_overflow:true,page_errors:errors}));
}finally{await browser.close()}})().catch(e=>{console.error(e);process.exit(1)});
