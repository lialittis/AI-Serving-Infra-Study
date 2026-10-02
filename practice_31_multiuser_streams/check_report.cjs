const {chromium}=require(process.env.P31_BROWSER_PACKAGE||'/tmp/inference-report-check/node_modules/playwright');
const path=require('path'),fs=require('fs');
(async()=>{
 const report=path.resolve(process.argv[2]);
 const browser=await chromium.launch({headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1450,height:1000},offline:true});
  const errors=[];page.on('pageerror',e=>errors.push(String(e)));page.on('requestfailed',r=>errors.push(r.url()));
  await page.goto('file://'+report);
  await page.waitForFunction(()=>window.P31,{timeout:60000});
  const counts=await page.evaluate(()=>({requests:P31.data.requests.length,tasks:(P31.data.tasks||[]).length,steps:(P31.data.steps||[]).length,status:P31.data.analysis_status}));
  if(!counts.requests)throw Error('No requests');
  if(counts.tasks){
   if(counts.status!=='passed')throw Error('Incomplete real attribution');
   const rid=await page.evaluate(()=>P31.data.requests[0].client_request_id);
   await page.selectOption('#request',rid);
   const opts=await page.locator('#step option').count();if(opts<2)throw Error('Missing request-to-step linkage');
   const step=await page.locator('#step option').nth(1).getAttribute('value');await page.selectOption('#step',step);
   await page.locator('#tasks tr').first().click();
   if(!(await page.locator('#detail').textContent()).includes('native_submission'))throw Error('Missing exact flow evidence');
   const shared=await page.evaluate(()=>P31.data.tasks.some(t=>t.requests.length>1));
   if(await page.evaluate(()=>P31.data.concurrency>1)){if(!shared)throw Error('Expected observed shared batch');}
  }
  await page.locator('#fit').click();await page.setViewportSize({width:760,height:900});
  const caseName=path.basename(path.dirname(path.dirname(report)));
  await page.screenshot({path:'/tmp/p31-report-'+caseName+'.png',fullPage:false});
  if(errors.length)throw Error(errors.join('\n'));
  const result={report,offline:true,request_step_task_linkage:counts.tasks>0,counts,errors};
  console.log(JSON.stringify(result,null,2));
  if(process.argv[3])fs.writeFileSync(process.argv[3],JSON.stringify(result,null,2)+'\n');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
