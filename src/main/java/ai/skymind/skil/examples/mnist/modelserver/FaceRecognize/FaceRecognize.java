package ai.skymind.skil.examples.mnist.modelserver.FaceRecognize;

import ai.skymind.skil.examples.mnist.modelserver.auth.Authorization;
import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.mashape.unirest.http.Unirest;
import com.mashape.unirest.http.exceptions.UnirestException;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.transform.ImageTransformProcess;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.serde.base64.Nd4jBase64;

import java.io.File;
import java.io.IOException;

public class FaceRecognize {
    static String ip = "localhost";
    static String port = "8080";
    static String userName = "";
    static String password = "";

    @Parameter(names="--endpoint", description="Endpoint for classification", required=false)
    private String skilInferenceEndpoint = "http://"+ip+":"+port+"/endpoints/exp1model1/model/pepperfacerecognizevgg16/default/";
    @Parameter(names="--input", description="image input file", required=false)
    private String inputImageFile = "/face/100.jpg";

    public void run() throws Exception, IOException {

        ImageTransformProcess imgTransformProcess = new ImageTransformProcess.Builder().seed(12345)
            .build();
        
        File imageFile = null;
        INDArray finalRecord = null;

        if ("blank".equals(inputImageFile)) {

            finalRecord = Nd4j.zeros( 1, 100*100 );

            System.out.println( "Generating blank test image ..." );

        } else {

//            imageFile = new File( inputImageFile );
            imageFile = new ClassPathResource(inputImageFile).getFile();

            if (!imageFile.exists() || !imageFile.isFile()) {
                System.err.format("unable to access file %s\n", inputImageFile);
                System.exit(2);
            } else {

                System.out.println( "Inference for: " + inputImageFile );

            }
            NativeImageLoader imageLoader = new NativeImageLoader(100,100, 1);
            INDArray imgNdarray = imageLoader.asMatrix(imageFile);
            System.out.print("shape:===>"+imgNdarray.shapeInfoToString());
            finalRecord = imgNdarray;
        }

        String imgBase64 = Nd4jBase64.base64String(finalRecord);

        System.out.println( imgBase64 );  

        System.out.println( "Finished image conversion" );

        skilClientGetImageInference( imgBase64 );

    }

    private void skilClientGetImageInference( String imgBase64 ) {

        Authorization auth = new Authorization(ip,port);
        String auth_token = auth.getAuthToken( "admin", "admin123" );

        System.out.println( "auth token: " + auth_token );

        try {

            String returnVal =
                    Unirest.post( skilInferenceEndpoint + "multiclassify" ) 
                            .header("accept", "application/json")
                            .header("Content-Type", "application/json")
                            .header( "Authorization", "Bearer " + auth_token)
                            .body(new JSONObject() //Using this because the field functions couldn't get translated to an acceptable json
                                    .put( "id", "some_id" )
                                    .put("prediction", new JSONObject().put("array", imgBase64))
                                    .toString())
                            .asJson()
                            .getBody().getObject().toString(); 


            System.out.println( "classification return: " + returnVal );

        } catch (UnirestException e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) throws Exception {
        FaceRecognize m = new FaceRecognize();

        JCommander.newBuilder()
          .addObject(m)
          .build()
          .parse(args);
        m.run();
    }
}
