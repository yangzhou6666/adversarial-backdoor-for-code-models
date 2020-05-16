package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.ProgramFeatures;
import org.apache.commons.lang3.StringUtils;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;

class ExtractFeaturesTask implements Callable<Void> {
    private final CommandLineValues m_CommandLineValues;
    private final Path filePath;
    private final String contents;
    private final String fromFile;

    public ExtractFeaturesTask(CommandLineValues commandLineValues, Path path) {
        m_CommandLineValues = commandLineValues;
        this.filePath = path;
        this.contents = null;
        this.fromFile = null;
    }

    public ExtractFeaturesTask(CommandLineValues commandLineValues, String contents, String fromFile) {
        m_CommandLineValues = commandLineValues;
        this.filePath = null;
        this.contents = contents;
        this.fromFile = fromFile;
    }

    @Override
    public Void call() {
        processFile();
        return null;
    }

    public void processFile() {
        ArrayList<ProgramFeatures> features;
        try {
            features = extractSingleFile();
        } catch (IOException e) {
            e.printStackTrace();
            return;
        }
        if (features == null) {
            return;
        }

        String toPrint = featuresToString(features);
        if (toPrint.length() > 0) {
            // TODO: Is this where we say what the (original) sha was?
            // To tie transformed samples back to originals
            System.out.println(toPrint);
        }
    }

    private static int countLines(String str){
        String[] lines = str.split("\r\n|\r|\n");
        return  lines.length;
    }

    private ArrayList<ProgramFeatures> extractSingleFile() throws IOException {
        String code;

        if (filePath != null) {
            if (m_CommandLineValues.MaxFileLength > 0 &&
                    Files.lines(filePath, Charset.defaultCharset()).count() > m_CommandLineValues.MaxFileLength) {
                return new ArrayList<>();
            }
            try {
                code = new String(Files.readAllBytes(filePath));
            } catch (IOException e) {
                e.printStackTrace();
                code = Common.EmptyString;
            }
        } else {
            if (m_CommandLineValues.MaxFileLength > 0 &&
                    countLines(contents) > m_CommandLineValues.MaxFileLength) {
                return new ArrayList<>();
            }
            code = contents;
        }

        try {
            FeatureExtractor featureExtractor = new FeatureExtractor(m_CommandLineValues);
            return featureExtractor.extractFeatures(code);
        } catch (Exception ex) {
            return new ArrayList<>();
        }
    }

    public String featuresToString(ArrayList<ProgramFeatures> features) {
        if (features == null || features.isEmpty()) {
            return Common.EmptyString;
        }

        List<String> methodsOutputs = new ArrayList<>();

        for (ProgramFeatures singleMethodFeatures : features) {
            StringBuilder builder = new StringBuilder();

            String toPrint = singleMethodFeatures.toString();
            if (m_CommandLineValues.PrettyPrint) {
                toPrint = toPrint.replace(" ", "\n\t");
            }
            builder.append(toPrint);


            methodsOutputs.add(fromFile + " " + builder.toString());

        }
        return StringUtils.join(methodsOutputs, "\n");
    }
}
