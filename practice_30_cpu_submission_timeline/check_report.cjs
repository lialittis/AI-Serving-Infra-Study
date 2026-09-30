const {chromium}=require(process.env.P30_BROWSER_PACKAGE||'/tmp/inference-report-check/node_modules/playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const browser=await chromium.launch({headless:true});
 const page=await browser.newPage({viewport:{width:1450,height:1000},offline:true});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()));
 await page.goto('file://'+path.join(__dirname,'report/index.html'));
 await page.waitForFunction(()=>window.p30&&document.querySelector('#step').options.length===64);
 if(await page.locator('#examples button').count()!==4)throw Error('Missing four examples');
 for(const name of ['输入 H2D','线性层','RoPE','token D2H 与等待']){
   await page.getByRole('button',{name,exact:true}).click();
   if(await page.locator('#timeline .flow').count()<3)throw Error('Missing exact flow path');
   if(!(await page.locator('#source').textContent()).includes('/vllm-workspace/'))throw Error('Missing installed source');
 }
 await page.locator('#phases button').filter({hasText:/^调度$/}).click();
 if(!(await page.locator('#source').textContent()).includes('BalanceScheduler'))throw Error('Actual scheduler missing');
 await page.locator('#timeline').scrollIntoViewIfNeeded();await page.screenshot({path:'/tmp/p30-timeline.png'});
 await page.selectOption('#step','0');if((await page.locator('#examples button').count())!==0)throw Error('decode examples on prefill');
 await page.selectOption('#step','63');await page.locator('#focus32').click();
 await page.setViewportSize({width:600,height:900});await page.locator('#timeline').scrollIntoViewIfNeeded();
 await page.screenshot({path:'/tmp/p30-narrow.png'});
 await page.goto('file://'+path.join(__dirname,'report/decode32.svg'));
 if(await page.locator('rect title').count()<1000)throw Error('Missing measured bars');
 if(errors.length)throw Error(errors.join('\n'));
 const out=path.join(__dirname,'results/validation');fs.mkdirSync(out,{recursive:true});
 fs.writeFileSync(path.join(out,'browser.json'),JSON.stringify({offline:true,steps:64,examples:4,
   source_excerpts:true,actual_scheduler:true,exact_flow_highlighting:true,narrow_view:true,standalone_svg:true,errors},null,2)+'\n');
 console.log('Offline report checks passed');await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
